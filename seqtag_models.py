import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchcrf import CRF


class TokenCharEmbedding(nn.Module):

    def __init__(self, token_emb, char_emb, char_hidden_dim):
        super(TokenCharEmbedding, self).__init__()
        self.token_emb = token_emb
        self.char_emb = char_emb
        self.char_lstm = nn.LSTM(input_size=self.char_emb.embedding_dim, hidden_size=char_hidden_dim, batch_first=True)

    def forward(self, token_chars, token_char_lengths):
        batch_size = token_chars.shape[0]
        token_seq_length = token_chars.shape[1]
        char_seq_length = token_chars.shape[2]
        token_seq = token_chars[:, :, 0, 0]
        char_seq = token_chars[:, :, :, 1]
        char_lengths = token_char_lengths[:, :, 1]
        embed_chars = self.char_emb(char_seq)
        char_inputs = embed_chars.view(batch_size * token_seq_length, char_seq_length, -1)
        char_outputs, char_hidden_state = self.char_lstm(char_inputs)
        char_outputs = char_outputs[torch.arange(char_outputs.shape[0]), char_lengths.view(-1) - 1]
        char_outputs = char_outputs.view(batch_size, token_seq_length, -1)
        embed_tokens = self.token_emb(token_seq)
        embed_tokens = torch.cat((embed_tokens, char_outputs), dim=2)
        return embed_tokens

    @property
    def embedding_dim(self):
        return self.token_emb.embedding_dim + self.char_lstm.hidden_size


class FixedSequenceClassifier(nn.Module):

    def __init__(self, input_emb, encoder, dropout, max_seq_len, num_classes):
        super(FixedSequenceClassifier, self).__init__()
        self.input_emb = input_emb
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList([nn.Linear(encoder.hidden_size, num_classes) for _ in range(max_seq_len)])

    @property
    def num_labels(self):
        return self.classifiers[0].out_features

    def forward(self, inputs, input_lengths):
        token_lengths = input_lengths[:, 0, 0]
        embed_tokens = self.input_emb(inputs, input_lengths)
        enc_tokens = self.encoder(embed_tokens, token_lengths)
        enc_tokens = self.dropout(enc_tokens)
        enc_tokens = torch.tanh(enc_tokens)
        # return self.classifiers(enc_tokens)
        return [classifier(enc_tokens) for classifier in self.classifiers]

    def loss(self, label_scores, gold_labels, label_masks):
        losses = []
        loss_fct = nn.CrossEntropyLoss()
        for i in range(len(label_scores)):
            labels = gold_labels[:, :, i].view(-1)[label_masks.view(-1)]
            scores = label_scores[i].view(-1, label_scores[i].shape[2])[label_masks.view(-1)]
            losses.append(loss_fct(scores, labels))
        return losses

    def decode(self, label_scores):
        labels = [torch.argmax(scores, dim=2) for scores in label_scores]
        return torch.stack(labels, dim=2)


class BatchEncoder(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(BatchEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_dim // 2, num_layers=num_layers, batch_first=True,
                           bidirectional=True, dropout=(dropout if num_layers > 1 else 0))

    def forward(self, embed_inputs, input_lengths):
        # https: // gist.github.com / HarshTrivedi / f4e7293e941b17d19058f6fb90ab0fec
        sorted_lengths, sorted_perm_idx = input_lengths.sort(0, descending=True)
        packed_seq = pack_padded_sequence(embed_inputs[sorted_perm_idx], sorted_lengths, batch_first=True)
        packed_outputs, _ = self.rnn(packed_seq)
        padded_output, seq_lengths = pad_packed_sequence(packed_outputs, batch_first=True, total_length=embed_inputs.shape[1])
        _, reflect_sorted_perm_idx = sorted_perm_idx.sort()
        return padded_output[reflect_sorted_perm_idx]

    @property
    def hidden_size(self):
        return self.rnn.hidden_size * 2


class SequenceStepDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, num_labels):
        super(SequenceStepDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=(dropout if num_layers > 1 else 0))
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, input_seq, hidden_state):
        output_seq, hidden_state = self.rnn(input_seq, hidden_state)
        # outputs = torch.tanh(outputs)
        # outputs = torch.relu(outputs)
        seq_scores = self.output(output_seq)
        return seq_scores, hidden_state

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)

    @property
    def num_labels(self):
        return self.output.out_features

    @property
    def num_layers(self):
        return self.rnn.num_layers


class Seq2SeqClassifier(nn.Module):

    def __init__(self, enc_emb, encoder, dec_emb, decoder, max_label_seq_len, sos, eot):
        super(Seq2SeqClassifier, self).__init__()
        self.enc_emb = enc_emb
        self.encoder = encoder
        self.dec_emb = dec_emb
        self.decoder = decoder
        self.max_label_seq_len = max_label_seq_len
        self.sos = sos
        self.eot = eot

    def forward(self, inputs, input_lengths, gold_labels=None):
        embed_inputs = self.enc_emb(inputs, input_lengths)
        dec_hidden_state = self.forward_encode(embed_inputs)
        return self.get_label_seq_scores(dec_hidden_state, input_lengths[:, 0, 0], embed_inputs, gold_labels)
        # scores = self.get_label_seq_scores(dec_hidden_state, input_lengths[:, 0, 0], embed_inputs, gold_labels)
        # return torch.stack(scores, dim=1)

    def forward_encode(self, embed_inputs):
        batch_size = embed_inputs.shape[0]
        enc_inputs, hidden_state = self.encoder(embed_inputs)
        enc_h = hidden_state[0].view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size)
        enc_c = hidden_state[1].view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size)
        dec_h = enc_h[-self.decoder.num_layers:].transpose(dim0=2, dim1=3)
        dec_h = dec_h.reshape(self.decoder.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_c = enc_c[-self.decoder.num_layers:].transpose(dim0=2, dim1=3)
        dec_c = dec_c.reshape(self.decoder.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_hidden_state = (dec_h, dec_c)
        return dec_hidden_state

    def get_label_seq_scores(self, hidden_state, token_lengths, embed_tokens, gold_labels):
        # I am using a tensor instead of just appending scores to a list because I want to keep track of the
        # labels in the analysis [N * 6 * 51] structure (N - number of tokens, 6 - number of morphemes in an analysis,
        # 51 - number of label scores)
        # scores = []
        scores = embed_tokens.new_full((embed_tokens.shape[0], embed_tokens.shape[1], self.max_label_seq_len,
                                        self.decoder.num_labels), fill_value=-1e10, requires_grad=False)
        scores[:, :, :, 0] = 0.0
        batch_size = embed_tokens.shape[0]
        embed_label = self.dec_emb(self.sos.repeat(batch_size).view(batch_size, 1))
        token_indices = torch.zeros_like(token_lengths)
        label_indices = torch.zeros_like(token_lengths)
        while torch.any(torch.lt(token_indices, token_lengths)):
            if embed_tokens is not None:
                index = token_indices.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, embed_tokens.shape[-1])
                embed_token = torch.gather(embed_tokens, 1, index)
                embed_label = torch.cat([embed_label, embed_token], dim=2)
            dec_scores, hidden_state = self.decoder(embed_label, hidden_state)
            # TODO: figure out why:
            # TODO: scores[0, token_indices, label_indices] = dec_scores
            # TODO: generates the following runtime error message:
            # TODO: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace
            # TODO: operation: [torch.LongTensor [1]] is at version 41; expected version 40 instead.
            scores[0, token_indices.item(), label_indices.item()] = dec_scores[0]
            # scores.append(dec_scores.squeeze(dim=1))
            if gold_labels is not None:
                index = token_indices.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, gold_labels.shape[-1])
                pred_label = torch.gather(gold_labels, 1, index)
                pred_label = pred_label[:, :, label_indices].squeeze(dim=1)
            else:
                pred_label = self.decoder.decode(dec_scores)
            token_length_mask = token_lengths == token_indices
            # <PAD> all predictions beyond sentence tokens
            pred_label[token_length_mask] = 0
            # Check for <EOT> or if we've reached max labels
            pred_label_mask = pred_label.squeeze(dim=1) == self.eot
            pred_label_mask |= label_indices == self.max_label_seq_len - 1
            # Advance tokens (if reached <EOT> or max labels)
            token_indices += pred_label_mask
            # Advance label if still in this token
            label_indices += ~pred_label_mask
            # Zero out labels that advanced to next token
            label_indices[pred_label_mask] = 0
            embed_label = self.dec_emb(pred_label)
        return scores

    def loss(self, label_scores, gold_labels, label_mask):
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(label_scores[label_mask], gold_labels[label_mask])

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=3)


class CrfClassifier(nn.Module):

    def __init__(self, classifier):
        super(CrfClassifier, self).__init__()
        self.classifier = classifier
        self.crf = CRF(classifier.num_classes, batch_first=True)

    def forward(self, *inputs):
        return self.classifier(inputs)

    def loss(self, label_scores, gold_labels, labels_mask):
        # classifier_loss = self.classifier.loss(label_scores, gold_labels, labels_mask)
        log_likelihood = self.crf(emissions=label_scores, tags=gold_labels, mask=labels_mask, reduction='token_mean')
        return -log_likelihood

    def decode(self, label_scores, mask=None):
        decoded_classes = self.crf.decode(emissions=label_scores, mask=mask)
        decoded_classes = [torch.tensor(t, dtype=torch.long) for t in decoded_classes]
        return pad_sequence(decoded_classes, batch_first=True, padding_value=0)
