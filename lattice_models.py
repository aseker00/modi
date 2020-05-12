import torch
import torch.nn as nn


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

    def forward(self, lattice, lattice_mask, inputs, input_lengths, gold_indices=None):
        embed_lattice = self.lattice_emb(lattice)
        embed_inputs = self.input_emb(inputs, input_lengths)
        # embed_lattice = self.get_embed_input_lattice(embed_lattice, embed_inputs)
        batch_size = embed_lattice.shape[0]
        enc_lattice, hidden_state = self.forward_lattice_encode(embed_lattice, lattice_mask)
        scores = self.get_lattice_pointers(enc_lattice, hidden_state, embed_lattice, embed_inputs, input_lengths, lattice_mask, gold_indices)
        scores = [torch.nn.utils.rnn.pad_sequence([s[i, 0] for s in scores], batch_first=True, padding_value=-1e10)
                  for i in range(batch_size)]
        return torch.stack(scores, dim=0)

    # Add the input token embedding to the lattice analysis embedding
    # Can be used as the input to the encoder instead of just using the lattice analysis embedding
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
            token_idx = len(scores)
            token_mask = packed_lattice_mask[:, 1] == token_idx
            enc_token_lattice = enc_lattice[:, token_mask]
            if embed_tokens is not None:
                embed_token = embed_tokens[:, token_idx].unsqueeze(dim=1)
                embed_analysis = torch.cat([embed_analysis, embed_token], dim=2)
            dec_scores, hidden_state = self.analysis_decoder(embed_analysis, hidden_state)
            dec_analysis_weights = self.analysis_attn(dec_scores, enc_token_lattice)
            scores.append(dec_analysis_weights)
            if gold_indices is not None:
                analysis_indices = gold_indices[:, token_idx].unsqueeze(dim=1)
                # Handle uninfused gold lattices
                missing_gold_indices_mask = analysis_indices[:, 0] == -1
                if missing_gold_indices_mask.any():
                    with torch.no_grad():
                        analysis_indices = self.decode(dec_analysis_weights)
                    analysis_indices[~missing_gold_indices_mask] = gold_indices[:, token_idx][~missing_gold_indices_mask].unsqueeze(dim=1)
            else:
                with torch.no_grad():
                    analysis_indices = self.decode(dec_analysis_weights)
            embed_analysis = embed_lattice[:, token_idx][:, analysis_indices[:, 0]]
        return scores

    def loss(self, lattice_scores, gold_indices, token_masks):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        return loss_fct(lattice_scores.squeeze(dim=0), gold_indices[token_masks])

    def decode(self, lattice_scores):
        return torch.argmax(lattice_scores, dim=2)
