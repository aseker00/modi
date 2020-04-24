import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchcrf import CRF
import numpy as np


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
        masked_scores = [torch.argmax(scores, dim=2) for scores in label_scores]
        return torch.stack(masked_scores, dim=2)


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
        scores = self.get_label_seq_scores(dec_hidden_state, input_lengths[:, 0, 0], embed_inputs, gold_labels)
        return torch.stack(scores, dim=1)

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

    def get_input_seq_scores(self, hidden_state, input_lengths, embed_inputs, gold_labels):
        scores = []
        batch_size = input_lengths.shape[0]
        # Initial <SOS> label
        pred_label = self.sos.repeat(batch_size).view(batch_size, 1)
        # embed_label = self.dec_emb(self.sos).repeat(batch_size).view(batch_size, 1, -1)
        # Keep track of current token index and analysis morpheme index being decoded
        input_indices = np.zeros(batch_size, dtype=np.int)
        label_indices = np.zeros(batch_size, dtype=np.int)
        # Stop when all current token indices have reached their input token lengths
        while np.any(np.less(input_indices, input_lengths[:, 0, 0].numpy())):
        # while np.any(np.less(input_indices, input_lengths.numpy())):
            embed_label = self.dec_emb(pred_label)
            # If embed inputs are available use the current token along with the previous label
            if embed_inputs is not None:
                embed_label = torch.cat([embed_label, embed_inputs[:, input_indices]], dim=2)
            # Decode current label
            dec_scores, hidden_state = self.decoder(embed_label, hidden_state)
            scores.append(dec_scores.squeeze(dim=1))
            # If gold label is available use it, otherwise use the decoded label
            if gold_labels is not None:
                pred_label = gold_labels[:, input_indices, label_indices]
            else:
                pred_label = self.decode(dec_scores)
            # All labels beyond token lengths get <PAD> (this is only relevant for batch_size > 1, where one sentence
            # may be done, but other sentences haven't reached their token lengths)
            pred_label[:, input_indices == input_lengths.numpy()] = 0
            # Get <EOT> mask, which is used as an indicator to increment token indices
            # TODO: I think there's a bug here because if not using gold_labels for next pred_label it might be that
            # TODO: the <EOT> is not present but we've reached max_label_seq_len so we need increment input index
            # TODO: in this case
            pred_label_mask = (pred_label.squeeze(dim=1) == self.eot).numpy()
            # TODO: So I think the label indices should be checked for max label seq before incrementing input indices
            pred_label_mask |= label_indices == self.max_label_seq_len - 1
            input_indices += pred_label_mask
            # Next we need to increment label indices and zero out label index if the token index incremented
            # TODO: do we need to increment input index if label_index == max_label_seq_len?
            # pred_label_mask |= label_indices == self.max_label_seq_len - 1
            # Increment label indices wherever we haven't reached <EOT> or max_label_seq_len
            label_indices += ~pred_label_mask
            # Zero out label indices wherever we have reached <EOT> or max_lebel_seq_len
            label_indices[pred_label_mask] = 0
            # embed_label = self.dec_emb(pred_label)
        return scores

    def get_label_seq_scores(self, hidden_state, token_lengths, embed_tokens, gold_labels):
        scores = []
        batch_size = embed_tokens.shape[0]
        token_seq_len = embed_tokens.shape[1]
        embed_label = self.dec_emb(self.sos.repeat(batch_size).view(batch_size, 1))
        token_indices = torch.zeros(batch_size, dtype=torch.long, requires_grad=False)
        label_indices = torch.zeros(batch_size, dtype=torch.long, requires_grad=False)
        token_ranges = torch.arange(token_seq_len, dtype=torch.long, requires_grad=False).repeat(batch_size, 1)
        while torch.any(torch.lt(token_indices, token_lengths)):
            token_index_mask = token_ranges == token_indices
            if embed_tokens is not None:
                embed_token = embed_tokens[token_index_mask].unsqueeze(dim=1)
                embed_label = torch.cat([embed_label, embed_token], dim=2)
            dec_scores, hidden_state = self.decoder(embed_label, hidden_state)
            scores.append(dec_scores.squeeze(dim=1))
            if gold_labels is not None:
                # It is OK to use label indices after we use the boolean token index mask because the mask generates a
                # cloned tensor separate from the gold labels tensor
                pred_label = gold_labels[token_index_mask][:, label_indices]
            else:
                pred_label = self.decode(dec_scores)
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

    def loss(self, label_scores, gold_labels):
        loss_fct = nn.CrossEntropyLoss()
        # masked_gold_labels = gold_labels.view(-1)[labels_mask.view(-1)]
        # masked_label_scores = label_scores.view(-1, label_scores.shape[2])[labels_mask.view(-1)]
        masked_gold_labels = gold_labels.view(-1)
        masked_label_scores = label_scores.view(-1, label_scores.shape[2])
        return loss_fct(masked_label_scores, masked_gold_labels)

    def decode(self, label_scores):
        return self.decoder.decode(label_scores)


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
