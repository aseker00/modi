import torch
import torch.nn as nn
# import numpy as np


class MorphEmbedding(nn.Module):

    def __init__(self, form_emb, lemma_emb, tag_emb, feats_emb, num_feats):
        super(MorphEmbedding, self).__init__()
        self.form_emb = form_emb
        self.lemma_emb = lemma_emb
        self.tag_emb = tag_emb
        self.feats_emb = feats_emb
        self.embedding_dim = sum([emb.embedding_dim for emb in [form_emb, lemma_emb, tag_emb]])
        self.embedding_dim += feats_emb.embedding_dim * num_feats

    def forward(self, lattices):
        forms = lattices[:, :, :, :, 0]
        lemmas = lattices[:, :, :, :, 1]
        tags = lattices[:, :, :, :, 2]
        feats = lattices[:, :, :, :, 3:]
        embedded_forms = self.form_emb(forms)
        embedded_lemmas = self.lemma_emb(lemmas)
        embedded_tags = self.tag_emb(tags)
        embedded_feats = self.feats_emb(feats)
        embedded_feats = embedded_feats.view(lattices.shape[0], lattices.shape[1], -1, embedded_feats.shape[2])
        embedded_feats = embedded_feats.mean(2)
        return torch.cat([embedded_forms, embedded_lemmas, embedded_tags, embedded_feats], dim=2)


class AnalysisEmbedding(MorphEmbedding):

    def __init__(self, form_emb, lemma_emb, tag_emb, feats_emb, num_feats):
        super(AnalysisEmbedding, self).__init__(form_emb, lemma_emb, tag_emb, feats_emb, num_feats)

    def forward(self, lattices):
        forms = lattices[:, :, :, :, 0]
        lemmas = lattices[:, :, :, :, 1]
        tags = lattices[:, :, :, :, 2]
        feats = lattices[:, :, :, :, 3:]
        embedded_forms = self.form_emb(forms).mean(dim=-2)
        embedded_lemmas = self.lemma_emb(lemmas).mean(dim=-2)
        embedded_tags = self.tag_emb(tags).mean(dim=-2)
        embedded_feats = self.feats_emb(feats).mean(dim=-3).view(feats.shape[0], feats.shape[1], feats.shape[2], -1)
        return torch.cat([embedded_forms, embedded_lemmas, embedded_tags, embedded_feats], dim=-1)


class SequenceStepAttention(nn.Module):

    def __init__(self, attn_type=None):
        super(SequenceStepAttention, self).__init__()
        self.attn_type = attn_type

    def forward(self, query, context):
        return torch.bmm(query, context.transpose(-2, -1))


class LatticeTokenPtrNet(nn.Module):

    def __init__(self, lattice_emb, input_emb, lattice_encoder, analysis_decoder, analysis_attn, sos):
        super(LatticeTokenPtrNet, self).__init__()
        self.lattice_emb = lattice_emb
        self.input_emb = input_emb
        self.lattice_encoder = lattice_encoder
        self.analysis_decoder = analysis_decoder
        self.analysis_attn = analysis_attn
        self.sos = sos

    def forward(self, lattice, analysis_lengths, inputs, input_lengths, gold_indices=None):
        embed_lattice = self.lattice_emb(lattice)
        embed_inputs = self.input_emb(inputs, input_lengths)
        # embed_lattice = self.get_embed_input_lattice(embed_lattice, embed_inputs)
        batch_size = embed_lattice.shape[0]
        input_seq_len = embed_lattice.shape[1]
        analysis_seq_len = embed_lattice.shape[2]
        lattice_mask_index = torch.arange(analysis_seq_len, dtype=torch.long).repeat(batch_size, input_seq_len, 1)
        lattice_mask = torch.lt(lattice_mask_index, analysis_lengths.unsqueeze(dim=2))
        # lattice_mask_index = np.tile(np.arange(analysis_seq_len, dtype=np.int), (batch_size, input_seq_len, 1))
        # lattice_mask = np.less(lattice_mask_index, analysis_lengths.numpy().reshape(batch_size, input_seq_len, 1))
        enc_lattice, hidden_state = self.forward_lattice_encode(embed_lattice, lattice_mask)
        scores = self.get_lattice_pointers2(enc_lattice, hidden_state, embed_lattice, embed_inputs, input_lengths, lattice_mask, gold_indices)
        scores = [torch.nn.utils.rnn.pad_sequence([s[i, 0] for s in scores], batch_first=True, padding_value=-1e10) for i in range(batch_size)]
        return torch.stack(scores, dim=0)

    def get_embed_input_lattice(self, embed_lattice, embed_inputs):
        embed_inputs = embed_inputs.repeat(1, 1, embed_lattice.shape[2])
        embed_inputs = embed_inputs.view(embed_lattice.shape[0], embed_lattice.shape[1], embed_lattice.shape[2], -1)
        return torch.cat([embed_lattice, embed_inputs], dim=-1)

    def forward_lattice_encode(self, embed_lattice, lattice_mask):
        batch_size = embed_lattice.shape[0]
        embed_lattice = embed_lattice[lattice_mask].unsqueeze(dim=0)
        enc_lattice, hidden_state = self.lattice_encoder(embed_lattice)
        enc_h = hidden_state[0].view(self.lattice_encoder.num_layers, 2, batch_size, self.lattice_encoder.hidden_size)
        enc_c = hidden_state[1].view(self.lattice_encoder.num_layers, 2, batch_size, self.lattice_encoder.hidden_size)
        dec_h = enc_h[-self.analysis_decoder.num_layers:].transpose(dim0=2, dim1=3)
        dec_h = dec_h.reshape(self.analysis_decoder.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_c = enc_c[-self.analysis_decoder.num_layers:].transpose(dim0=2, dim1=3)
        dec_c = dec_c.reshape(self.analysis_decoder.num_layers, -1, batch_size).transpose(dim0=1, dim1=2).contiguous()
        dec_hidden_state = (dec_h, dec_c)
        return enc_lattice, dec_hidden_state

    def get_lattice_pointers(self, enc_lattice, hidden_state, embed_lattice, embed_tokens, token_lengths, lattice_mask, gold_indices):
        scores = []
        batch_size = embed_lattice.shape[0]
        embed_analysis = self.lattice_emb(self.sos.repeat(batch_size).view(batch_size, 1, 1, 1, -1)).view(batch_size, 1, -1)
        packed_lattice_mask = lattice_mask.nonzero()
        while torch.any(len(scores) < token_lengths[:, 0, 0]):
            cur_token_idx = len(scores)
            token_mask = packed_lattice_mask[:, 1] == cur_token_idx
            enc_token_lattice = enc_lattice[:, token_mask]
            if embed_tokens is not None:
                embed_token = embed_tokens[:, cur_token_idx].unsqueeze(dim=1)
                embed_analysis = torch.cat([embed_analysis, embed_token], dim=2)
            dec_scores, hidden_state = self.analysis_decoder(embed_analysis, hidden_state)
            dec_analysis_weights = self.analysis_attn(dec_scores, enc_token_lattice)
            scores.append(dec_analysis_weights)
            if gold_indices is not None:
                pred_analysis_indices = gold_indices[:, cur_token_idx].unsqueeze(dim=1)
            else:
                pred_analysis_indices = self.decode(dec_analysis_weights)
            embed_analysis = embed_lattice[:, cur_token_idx][:, pred_analysis_indices[:, 0]]
        return scores

    def get_lattice_pointers2(self, enc_lattice, hidden_state, embed_lattice, embed_tokens, token_lengths, lattice_mask, gold_indices):
        scores = []
        batch_size = embed_lattice.shape[0]
        token_seq_len = embed_lattice.shape[1]
        embed_analysis = self.lattice_emb(self.sos.repeat(batch_size).view(batch_size, 1, 1, 1, -1)).view(batch_size, 1, -1)
        token_indices = torch.zeros(batch_size, dtype=torch.long, requires_grad=False)
        token_ranges = torch.arange(token_seq_len, dtype=torch.long, requires_grad=False).repeat(batch_size, 1)
        packed_lattice_mask = lattice_mask.nonzero()
        while torch.any(torch.lt(token_indices, token_lengths[:, 0, 0])):
            lattice_token_mask = packed_lattice_mask[:, 1] == token_indices
            enc_token_lattice = enc_lattice[:, lattice_token_mask]
            token_index_mask = token_ranges == token_indices
            if embed_tokens is not None:
                embed_token = embed_tokens[token_index_mask].unsqueeze(dim=1)
                embed_analysis = torch.cat([embed_analysis, embed_token], dim=2)
            dec_scores, hidden_state = self.analysis_decoder(embed_analysis, hidden_state)
            dec_analysis_weights = self.analysis_attn(dec_scores, enc_token_lattice)
            scores.append(dec_analysis_weights)
            if gold_indices is not None:
                pred_analysis_indices = gold_indices[token_index_mask].unsqueeze(dim=1)
            else:
                pred_analysis_indices = self.decode(dec_analysis_weights)
            # There is a question here: when using a boolean mask which generates a cloned tensor
            # separate from the original embed_lattice, doesn't that screw up the backpropagation from reaching
            # the lattice embedding (fasttext form and lemma + tag + feats embedding)?
            embed_analysis = embed_lattice[token_index_mask][:, pred_analysis_indices[:, 0]]
            token_indices += 1
        return scores

    def loss(self, lattice_scores, gold_indices, token_masks):
        loss_fct = nn.CrossEntropyLoss()
        # if gold_indices.nelement() == 0:
        #     return pred_indices
        missing_gold_indices = gold_indices[token_masks] == -1
        if missing_gold_indices.any():
            missing_gold_indices = missing_gold_indices[gold_indices[token_masks]]
            loss = loss_fct(lattice_scores[~missing_gold_indices], gold_indices[~missing_gold_indices])
        else:
            loss = loss_fct(lattice_scores.view(lattice_scores.shape[0] * lattice_scores.shape[1], -1), gold_indices[token_masks])
        return loss

    def decode(self, lattice_scores):
        return torch.argmax(lattice_scores, dim=2)
