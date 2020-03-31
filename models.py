import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchcrf import CRF


class TokenCharRNNEmbedding(nn.Module):

    def __init__(self, char_emb, hidden_dim, num_layers, dropout):
        super(TokenCharRNNEmbedding, self).__init__()
        self.char_emb = char_emb
        self.char_lstm = nn.LSTM(input_size=self.char_emb.embedding_dim, hidden_size=hidden_dim // 2, batch_first=True,
                                 bidirectional=True, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0))

    def forward(self, char_seq, char_lengths):
        batch_size = char_seq.shape[0]
        token_seq_length = char_seq.shape[1]
        char_seq_length = char_seq.shape[2]
        embed_chars = self.char_emb(char_seq)
        char_inputs = embed_chars.view(batch_size * token_seq_length, char_seq_length, -1)
        char_outputs, char_hidden_state = self.char_lstm(char_inputs)
        char_outputs = char_outputs[torch.arange(char_outputs.size(0)), char_lengths.view(-1) - 1]
        return char_outputs.view(batch_size, token_seq_length, -1)

    @property
    def embedding_dim(self):
        return self.char_lstm.hidden_size * 2


class TokenEmbedding(nn.Module):

    def __init__(self, token_emb, token_char_emb, dropout):
        super(TokenEmbedding, self).__init__()
        self.token_emb = token_emb
        self.dropout = nn.Dropout(dropout)
        self.token_char_emb = token_char_emb

    def forward(self, token_seq, char_seq, char_lengths):
        embed_tokens = self.token_emb(token_seq)
        embed_chars = self.token_char_emb(char_seq, char_lengths)
        embed_tokens = torch.cat([embed_tokens, embed_chars], dim=2)
        embed_tokens = self.dropout(embed_tokens)
        return embed_tokens

    @property
    def embedding_dim(self):
        return self.token_emb.embedding_dim + self.token_char_emb.embedding_dim


class TokenRNN(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(TokenRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_dim // 2, num_layers=num_layers, batch_first=True,
                           bidirectional=True, dropout=(dropout if num_layers > 1 else 0))

    def forward(self, embed_token_seq, token_lengths):
        # https: // gist.github.com / HarshTrivedi / f4e7293e941b17d19058f6fb90ab0fec
        sorted_lengths, sorted_perm_idx = token_lengths.sort(0, descending=True)
        packed_seq = pack_padded_sequence(embed_token_seq[sorted_perm_idx], sorted_lengths, batch_first=True)
        packed_outputs, packed_hidden_state = self.rnn(packed_seq)
        padded_output, seq_lengths = pad_packed_sequence(packed_outputs, batch_first=True, total_length=embed_token_seq.shape[1])
        _, reflect_sorted_perm_idx = sorted_perm_idx.sort()
        token_output = padded_output[reflect_sorted_perm_idx]
        hidden_state = tuple(torch.stack([layer[reflect_sorted_perm_idx] for layer in hs], dim=0) for hs in packed_hidden_state)
        return token_output, hidden_state

    @property
    def hidden_size(self):
        return self.rnn.hidden_size * 2


class TokenClassifier(nn.Module):

    def __init__(self, token_emb, token_rnn, dropout, num_classes):
        super(TokenClassifier, self).__init__()
        self.token_emb = token_emb
        self.token_rnn = token_rnn
        self.dropout = nn.Dropout(dropout)
        self.pref_output = nn.Linear(token_rnn.hidden_size, num_classes)
        self.host_output = nn.Linear(token_rnn.hidden_size, num_classes)
        self.suff_output = nn.Linear(token_rnn.hidden_size, num_classes)

    @property
    def num_labels(self):
        return self.host_output.out_features

    def forward(self, input_seq, input_lengths):
        embed_input_seq = self.token_emb(*input_seq)
        outputs, _ = self.token_rnn(embed_input_seq, input_lengths)
        outputs = self.dropout(outputs)
        outputs = torch.tanh(outputs)
        pref_scores = self.pref_output(outputs)
        host_scores = self.host_output(outputs)
        suff_scores = self.suff_output(outputs)
        # tag_scores = torch.stack((pref_tag_scores, host_tag_scores, suff_tag_scores), dim=1)
        # tag_scores = (tag_scores.view(tag_scores.shape[0], tag_scores.shape[1] * tag_scores.shape[2], -1)
        #               .transpose(dim0=1, dim1=2).contiguous()
        #               .view(tag_scores.shape[0], tag_scores.shape[1] * tag_scores.shape[2], -1))
        return pref_scores, host_scores, suff_scores

    def loss(self, label_scores, gold_labels, mask):
        loss_fct = nn.CrossEntropyLoss()
        masked_gold_pref_labels = gold_labels[:, :, 0].view(-1)[mask.view(-1)]
        masked_gold_host_labels = gold_labels[:, :, 1].view(-1)[mask.view(-1)]
        masked_gold_suff_labels = gold_labels[:, :, 2].view(-1)[mask.view(-1)]
        masked_pref_label_scores = label_scores[0].view(-1, label_scores[0].shape[2])[mask.view(-1)]
        masked_host_label_scores = label_scores[1].view(-1, label_scores[1].shape[2])[mask.view(-1)]
        masked_suff_label_scores = label_scores[2].view(-1, label_scores[2].shape[2])[mask.view(-1)]
        pref_loss = loss_fct(masked_pref_label_scores, masked_gold_pref_labels)
        host_loss = loss_fct(masked_host_label_scores, masked_gold_host_labels)
        suff_loss = loss_fct(masked_suff_label_scores, masked_gold_suff_labels)
        return pref_loss, host_loss, suff_loss

    def decode(self, label_scores):
        pref_labels = torch.argmax(label_scores[0], dim=2)
        host_labels = torch.argmax(label_scores[1], dim=2)
        suff_labels = torch.argmax(label_scores[2], dim=2)
        return torch.stack([pref_labels, host_labels, suff_labels], dim=2)


class TokenMorphSeqClassifier(nn.Module):

    def __init__(self, token_classifier):
        super(TokenMorphSeqClassifier, self).__init__()
        self.token_classifier = token_classifier
        self.crf = CRF(token_classifier.num_labels, batch_first=True)

    def forward(self, input_seq, input_lengths):
        return self.token_classifier(input_seq, input_lengths)

    def loss_crf(self, label_scores, gold_labels, mask):
        log_likelihood = self.crf(emissions=label_scores, tags=gold_labels, mask=mask, reduction='token_mean')
        return -log_likelihood

    def decode_crf(self, label_scores, mask=None):
        decoded_classes = self.crf.decode(emissions=label_scores, mask=mask)
        decoded_classes = [torch.tensor(t, dtype=torch.long) for t in decoded_classes]
        return pad_sequence(decoded_classes, batch_first=True, padding_value=0)


class MorphemeDecoder(nn.Module):

    def __init__(self, rnn, dropout, num_tags, et_tag_id):
        super(MorphemeDecoder, self).__init__()
        self.rnn = rnn
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(rnn.hidden_size, num_tags)
        self.et_tag_id = et_tag_id

    def forward(self, embed_input_seq, hidden_state):
        outputs, hidden_state = self.rnn(embed_input_seq, hidden_state)
        outputs = self.dropout(outputs)
        scores = self.output(outputs)
        return scores, hidden_state

    def loss(self, label_scores, gold_labels, mask):
        loss_fct = nn.CrossEntropyLoss()
        masked_gold_labels = gold_labels.view(-1)[mask.view(-1)]
        masked_label_scores = label_scores.view(-1, label_scores.shape[2])[mask.view(-1)]
        return loss_fct(masked_label_scores, masked_gold_labels)

    def decode(self, label_scores):
        return torch.argmax(label_scores, dim=2)

    @property
    def num_tags(self):
        return self.output.out_features


class Seq2SeqClassifier(nn.Module):

    def __init__(self, enc_emb, encoder, dec_emb, decoder, device=None):
        super(Seq2SeqClassifier, self).__init__()
        self.enc_emb = enc_emb
        self.encoder = encoder
        self.dec_emb = dec_emb
        self.decoder = decoder
        self.crf = CRF(decoder.num_tags, batch_first=True)
        self.device = device

    def forward(self, token_seq, token_char_seq, token_char_lengths, token_lengths, max_token_tags_num, gold_tag_seq=None):
        embed_tokens = self.enc_emb(token_seq, token_char_seq, token_char_lengths)
        enc_tokens, enc_hidden_state = self.encoder(embed_tokens, token_lengths)
        batch_size = embed_tokens.shape[0]
        enc_h = enc_hidden_state[0].view(self.encoder.rnn.num_layers, 2, batch_size, self.encoder.rnn.hidden_size)
        enc_c = enc_hidden_state[1].view(self.encoder.rnn.num_layers, 2, batch_size, self.encoder.rnn.hidden_size)
        dec_h = enc_h[-self.decoder.rnn.num_layers:].transpose(dim0=2, dim1=3).reshape(self.decoder.rnn.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_c = enc_c[-self.decoder.rnn.num_layers:].transpose(dim0=2, dim1=3).reshape(self.decoder.rnn.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_hidden_state = (dec_h, dec_c)
        if self.decoder.rnn.input_size == self.dec_emb.embedding_dim:
            tag_scores = self._forward_tag_decoding(dec_hidden_state, token_lengths, max_token_tags_num * token_seq.shape[1], gold_tag_seq)
        else:
            tag_scores = self._forward_tag_token_decoding(embed_tokens, dec_hidden_state, token_lengths, max_token_tags_num * token_seq.shape[1], gold_tag_seq)
        return torch.stack(tag_scores, dim=1)

    def _forward_tag_decoding(self, dec_hidden_state, token_lengths, max_tag_seq_len, gold_tag_seq):
        tag_scores = []
        batch_size = token_lengths.shape[0]
        embed_tag = self.dec_emb(torch.ones((batch_size, 1), dtype=torch.long, device=self.device))
        cur_token_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(max_tag_seq_len):
            dec_tag_scores, dec_hidden_state = self.decoder(embed_tag, dec_hidden_state)
            tag_scores.append(dec_tag_scores.squeeze(dim=1))
            if gold_tag_seq is None:
                pred_tag = self.decode(dec_tag_scores)
            else:
                pred_tag = gold_tag_seq[:, i].unsqueeze(dim=1)
            pred_et_tag_mask = pred_tag.squeeze(dim=1) == self.decoder.et_tag_id
            token_idx_mask = cur_token_idx < token_lengths
            pred_et_tag_mask = pred_et_tag_mask & token_idx_mask
            cur_token_idx += pred_et_tag_mask
            if torch.all(cur_token_idx == token_lengths):
                break
            embed_tag = self.dec_emb(pred_tag)
        return tag_scores

    def _forward_tag_token_decoding(self, embed_tokens, dec_hidden_state, token_lengths, max_tag_seq_len, gold_tag_seq):
        tag_scores = []
        batch_size = token_lengths.shape[0]
        embed_tag = self.dec_emb(torch.ones((batch_size, 1), dtype=torch.long, device=self.device))
        cur_token_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(max_tag_seq_len):
            embed_token = [t[idx] if idx < t.shape[0] else t[-1] for t, idx in zip(embed_tokens, cur_token_idx)]
            embed_token = torch.stack(embed_token, dim=0).unsqueeze(dim=1)
            embed_input = torch.cat([embed_token, embed_tag], dim=2)
            dec_tag_scores, dec_hidden_state = self.decoder(embed_input, dec_hidden_state)
            tag_scores.append(dec_tag_scores.squeeze(dim=1))
            if gold_tag_seq is None:
                pred_tag = self.decode(dec_tag_scores)
            else:
                pred_tag = gold_tag_seq[:, i].unsqueeze(dim=1)
            pred_et_tag_mask = pred_tag.squeeze(dim=1) == self.decoder.et_tag_id
            token_idx_mask = cur_token_idx < token_lengths
            pred_et_tag_mask = pred_et_tag_mask & token_idx_mask
            cur_token_idx += pred_et_tag_mask
            if torch.all(cur_token_idx == token_lengths):
                break
            embed_tag = self.dec_emb(pred_tag)
        return tag_scores

    def loss(self, label_scores, gold_labels, mask):
        return self.decoder.loss(label_scores, gold_labels, mask)

    def decode(self, label_scores):
        return self.decoder.decode(label_scores)

    def loss_crf(self, label_scores, gold_labels, mask):
        log_likelihood = self.crf(emissions=label_scores, tags=gold_labels, mask=mask, reduction='token_mean')
        return -log_likelihood

    def decode_crf(self, label_scores, mask):
        decoded_classes = self.crf.decode(emissions=label_scores, mask=mask)
        decoded_classes = [torch.tensor(t, dtype=torch.long) for t in decoded_classes]
        return pad_sequence(decoded_classes, batch_first=True, padding_value=0)
