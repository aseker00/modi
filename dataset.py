from collections import Counter, defaultdict

from pathlib import Path

import fasttext_emb as ft
import treebank as tb
import numpy as np
import pandas as pd
import csv
import os

_to_form_vec = np.vectorize(lambda x, vocab: vocab['forms'][x])
_to_lemma_vec = np.vectorize(lambda x, vocab: vocab['lemmas'][x])
_to_tag_vec = np.vectorize(lambda x, vocab: vocab['tags'][x])
_to_feat_vec = np.vectorize(lambda x, vocab: vocab['feats'][x])
_to_token_vec = np.vectorize(lambda x, vocab: vocab['tokens'][x])
_to_form_id_vec = np.vectorize(lambda x, vocab: vocab['form2id'][x])
_to_lemma_id_vec = np.vectorize(lambda x, vocab: vocab['lemma2id'][x])
_to_tag_id_vec = np.vectorize(lambda x, vocab: vocab['tag2id'][x])
_to_feat_id_vec = np.vectorize(lambda x, vocab: vocab['feat2id'][x])
_to_token_id_vec = np.vectorize(lambda x, vocab: vocab['token2id'][x])
_get_multi_tags_len = np.vectorize(lambda x: len(x.split('-')))
_get_feats_len = np.vectorize(lambda x: len(x.split('|')))


def _tag_ids_to_token_lattice(tag_ids, token_mask, vocab, feats_to_str):
    token_tag_ids = tag_ids[token_mask]
    # First tag in each token must have a value (non <XXX> tag)
    if np.any(token_tag_ids[:, 0] < vocab['tag2id']['_']):
        token_tag_ids[:, 0][token_tag_ids[:, 0] < vocab['tag2id']['_']] = vocab['tag2id']['_']
    token_tag_ids[token_tag_ids < vocab['tag2id']['_']] = vocab['tag2id']['<PAD>']
    token_form_ids = np.zeros_like(token_tag_ids)
    token_lemma_ids = np.zeros_like(token_tag_ids)
    token_feat_ids = np.zeros_like(token_tag_ids)
    token_form_ids[token_tag_ids != vocab['form2id']['<PAD>']] = vocab['form2id']['_']
    token_lemma_ids[token_tag_ids != vocab['lemma2id']['<PAD>']] = vocab['lemma2id']['_']
    token_feat_ids[token_tag_ids != vocab['feats2id']['<PAD>']] = vocab['feats2id']['_']
    token_forms = _to_form_vec(token_form_ids, vocab)
    token_lemmas = _to_lemma_vec(token_lemma_ids, vocab)
    token_tags = _to_tag_vec(token_tag_ids, vocab)
    token_feats_str = feats_to_str(_to_feat_vec(token_feat_ids, vocab))
    return np.stack([token_forms, token_lemmas, token_tags, token_feats_str], axis=1)


def _lattice_ids_to_token_lattice(token_lattice_ids, data_vocab, feats_to_str):
    token_form_ids = token_lattice_ids[:, :, 0]
    token_lemma_ids = token_lattice_ids[:, :, 1]
    token_tag_ids = token_lattice_ids[:, :, 2]
    token_feat_ids = token_lattice_ids[:, :, 3:]
    token_forms = _to_form_vec(token_form_ids, data_vocab)
    token_lemmas = _to_lemma_vec(token_lemma_ids, data_vocab)
    token_tags = _to_tag_vec(token_tag_ids, data_vocab)
    token_feats_str = feats_to_str(_to_feat_vec(token_feat_ids, data_vocab))
    return np.stack([token_forms, token_lemmas, token_tags, token_feats_str], axis=1)


def _lattice_data_to_tags(lattice_df):
    values = [x[1].tag.values for x in lattice_df.groupby('token_id')]
    max_len = max([len(a) for a in values])
    tags = np.full_like(lattice_df.tag.values, '<PAD>', dtype=object, shape=(len(values), max_len))
    for i, a in enumerate(values):
        tags[i, :len(a)] = a
    return tags


def _concat_data(lattices):
    for i, df in enumerate(lattices):
        df.insert(0, 'sent_id', i + 1)
    return pd.concat(lattices)


def _to_data(tokens, lattices):
    column_names = ['from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'analysis_id',
                    'morpheme_id']
    token_forms = lattices[:, 0, :]
    token_lemmas = lattices[:, 1, :]
    token_tags = lattices[:, 2, :]
    token_feats = lattices[:, 3, :]
    rows = []
    token_indices, morpheme_indices = (token_tags != '<PAD>').nonzero()
    for i, (token_idx, morpheme_idx) in enumerate(zip(token_indices, morpheme_indices)):
        from_node_id = i
        to_node_id = i + 1
        form = token_forms[token_idx, morpheme_idx]
        lemma = token_lemmas[token_idx, morpheme_idx]
        tag = token_tags[token_idx, morpheme_idx]
        feat = token_feats[token_idx, morpheme_idx]
        token = tokens[token_idx]
        row = [from_node_id, to_node_id, form, lemma, tag, feat, token_idx + 1, token, 0, morpheme_idx]
        rows.append(row)
    return pd.DataFrame(rows, columns=column_names)


def _tag_eval(gold_df, pred_df):
    gold_gb = gold_df.groupby([gold_df.sent_id, gold_df.token_id])
    pred_gb = pred_df.groupby([pred_df.sent_id, pred_df.token_id])
    gold_counts, pred_counts, intersection_counts = 0, 0, 0
    for (sent_id, token_id), gold in sorted(gold_gb):
        pred = pred_gb.get_group((sent_id, token_id))
        gold_count, pred_count = Counter(gold.tag.tolist()), Counter(pred.tag.tolist())
        intersection_count = gold_count & pred_count
        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())
    precision = intersection_counts / pred_counts if pred_counts else 0.0
    recall = intersection_counts / gold_counts if gold_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _seg_tag_eval(gold_df, pred_df):
    gold_gb = gold_df.groupby([gold_df.sent_id, gold_df.token_id])
    pred_gb = pred_df.groupby([pred_df.sent_id, pred_df.token_id])
    gold_counts, pred_counts, intersection_counts = 0, 0, 0
    for (sent_id, token_id), gold in sorted(gold_gb):
        pred = pred_gb.get_group((sent_id, token_id))
        gold_seg_tags = list(zip(gold.tag.tolist(), gold.form.tolist()))
        pred_seg_tags = list(zip(pred.tag.tolist(), pred.form.tolist()))
        gold_count, pred_count = Counter(gold_seg_tags), Counter(pred_seg_tags)
        intersection_count = gold_count & pred_count
        gold_counts += sum(gold_count.values())
        pred_counts += sum(pred_count.values())
        intersection_counts += sum(intersection_count.values())
    precision = intersection_counts / pred_counts if pred_counts else 0.0
    recall = intersection_counts / gold_counts if gold_counts else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _split_feats(feats):
    feat_set = set()
    for f in feats:
        fmap = defaultdict(list)
        for ff in f.split('|'):
            # for fff in ff.split('-'):
            if ff == '_':
                continue
            name, value = ff.split('=')
            fmap[name].append(value)
        for name in fmap:
            value = ''.join(fmap[name])
            feat_set.add(f'{name}={value}')
    return feat_set


def _to_vocab(tokens, chars, forms, lemmas, tags, feats):
    tokens = ['<PAD>'] + sorted(tokens)
    chars = ['<PAD>'] + sorted(list(chars))
    for m in [forms, lemmas, tags, feats]:
        if '_' in m:
            m.remove('_')
    forms = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(forms))
    lemmas = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(lemmas))
    tags = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(tags))
    feats_str = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(feats))
    feat_set = _split_feats(feats)
    feats = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(feat_set)
    token2id = {v: i for i, v in enumerate(tokens)}
    char2id = {v: i for i, v in enumerate(chars)}
    form2id = {v: i for i, v in enumerate(forms)}
    lemma2id = {v: i for i, v in enumerate(lemmas)}
    tag2id = {v: i for i, v in enumerate(tags)}
    feats2id = {v: i for i, v in enumerate(feats)}
    feats_str2id = {v: i for i, v in enumerate(feats_str)}
    return {'tokens': tokens, 'token2id': token2id, 'chars': chars, 'char2id': char2id,
            'forms': forms, 'form2id': form2id, 'lemmas': lemmas, 'lemma2id': lemma2id,
            'tags': tags, 'tag2id': tag2id, 'feats': feats, 'feats2id': feats2id,
            'feats_str': feats_str, 'feats_str2id': feats_str2id}


def _get_vocab(lattices_dataset):
    tokens, forms, lemmas, tags, feats = set(), set(), set(), set(), set()
    for partition_type in lattices_dataset:
        for ldf in lattices_dataset[partition_type]:
            tokens.update(set(ldf.token.astype(str)))
            forms.update(set(ldf.form.astype(str)))
            lemmas.update(set(ldf.lemma.astype(str)))
            tags.update(set(ldf.tag.astype(str)))
            feats.update(set(ldf.feats.astype(str)))
    chars = set([c for w in list(tokens) + list(forms) + list(lemmas) for c in w])
    return _to_vocab(tokens, chars, forms, lemmas, tags, feats)


def _get_vocabs_union(dv1, dv2):
    tokens = set(dv1['tokens'] + dv2['tokens']).difference({'<PAD>', '<SOS>', '<EOT>'})
    forms = set(dv1['forms'] + dv2['forms']).difference({'<PAD>', '<SOS>', '<EOT>'})
    lemmas = set(dv1['lemmas'] + dv2['lemmas']).difference({'<PAD>', '<SOS>', '<EOT>'})
    tags = set(dv1['tags'] + dv2['tags']).difference({'<PAD>', '<SOS>', '<EOT>'})
    feats = set(dv1['feats_str'] + dv2['feats_str']).difference({'<PAD>', '<SOS>', '<EOT>'})
    chars = set([c for w in list(tokens) + list(forms) + list(lemmas) for c in w])
    return _to_vocab(tokens, chars, forms, lemmas, tags, feats)


def _save_vocab_files(data_vocab_dir_path, data_vocab):
    os.makedirs(data_vocab_dir_path, exist_ok=True)
    for key in ['tokens', 'chars', 'forms', 'lemmas', 'tags', 'feats', 'feats_str']:
        data_vocab_file_path = data_vocab_dir_path / f'{key}.txt'
        with open(str(data_vocab_file_path), 'w') as f:
            f.write('\n'.join(data_vocab[key]))
        print(f'{key} vocab size: {len(data_vocab[key])}')


def _load_vocab_entries(data_vocab_dir_path):
    data_vocab = {}
    keys = {'tokens': 'token2id', 'chars': 'char2id', 'forms': 'form2id', 'lemmas': 'lemma2id', 'tags': 'tag2id',
            'feats': 'feats2id', 'feats_str': 'feats_str2id'}
    for key in keys:
        data_vocab_file_path = data_vocab_dir_path / f'{key}.txt'
        with open(str(data_vocab_file_path)) as f:
            entries = [line.strip() for line in f.readlines()]
            entry2ids = {v: k for k, v in enumerate(entries)}
            data_vocab[key] = entries
            data_vocab[keys[key]] = entry2ids
        print(f'{key} vocab size: {len(data_vocab[key])}')
    return data_vocab


def _load_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab):
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    return (ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, data_vocab['chars']),
            ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, data_vocab['tokens']))


def _load_morpheme_ft_emb(vocab_dir_path, ft_model_path, data_vocab):
    chars_emb, tokens_emb = _load_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)
    forms_vec_file_path = vocab_dir_path / 'forms.vec'
    lemmas_vec_file_path = vocab_dir_path / 'lemmas.vec'
    return (chars_emb, tokens_emb,
            ft.load_embedding_weight_matrix(ft_model_path, forms_vec_file_path, data_vocab['forms']),
            ft.load_embedding_weight_matrix(ft_model_path, lemmas_vec_file_path, data_vocab['lemmas']))


def _save_token_ft_emb_files(vocab_dir_path, ft_model_path, data_vocab):
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    if chars_vec_file_path.exists():
        chars_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, data_vocab['chars'])
    if tokens_vec_file_path.exists():
        tokens_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, data_vocab['tokens'])


def _save_morpheme_ft_emb_files(vocab_dir_path, ft_model_path, data_vocab):
    ft.ft_model = None
    _save_token_ft_emb_files(vocab_dir_path, ft_model_path, data_vocab)
    forms_vec_file_path = vocab_dir_path / 'forms.vec'
    lemmas_vec_file_path = vocab_dir_path / 'lemmas.vec'
    if forms_vec_file_path.exists():
        forms_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, forms_vec_file_path, data_vocab['forms'])
    if lemmas_vec_file_path.exists():
        lemmas_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, lemmas_vec_file_path, data_vocab['lemmas'])


def _to_tokens_row_values(lattice_data_row, data_vocab, char_ids):
    sent_idx = lattice_data_row.sent_id
    token_idx = lattice_data_row.token_id
    if lattice_data_row.token in char_ids:
        token_id, token_char_ids = char_ids[str(lattice_data_row.token)]
    else:
        token_id = data_vocab['token2id'][str(lattice_data_row.token)]
        token_char_ids = [data_vocab['char2id'][c] for c in str(lattice_data_row.token)]
        char_ids[str(lattice_data_row.token)] = (token_id, token_char_ids)
    return [[sent_idx, token_idx, i + 1, token_id, char_id]
            for i, char_id in enumerate(token_char_ids)]


def _to_forms_row_values(lattice_data_row, sent_lengths, data_vocab, char_ids):
    sent_idx = lattice_data_row.sent_id
    segment_idx = lattice_data_row.Index + 1 - sent_lengths[sent_idx]
    if lattice_data_row.form in char_ids:
        form_id, form_char_ids = char_ids[str(lattice_data_row.form)]
    else:
        form_id = data_vocab['form2id'][str(lattice_data_row.form)]
        form_char_ids = [data_vocab['char2id'][c] for c in str(lattice_data_row.form)]
        char_ids[str(lattice_data_row.form)] = (form_id, form_char_ids)
    return [[sent_idx, segment_idx, i + 1, form_id, char_id]
            for i, char_id in enumerate(form_char_ids)]


def _to_lattice_row_values(lattice_data_row, max_num_feats, data_vocab):
    sent_idx = lattice_data_row.sent_id
    token_idx = lattice_data_row.token_id
    analysis_id = lattice_data_row.analysis_id
    morpheme_id = lattice_data_row.morpheme_id
    form_id = data_vocab['form2id'][str(lattice_data_row.form)]
    lemma_id = data_vocab['lemma2id'][str(lattice_data_row.lemma)]
    tag_id = data_vocab['tag2id'][str(lattice_data_row.tag)]
    feat_ids = [data_vocab['feats2id'][f] for f in str(lattice_data_row.feats).split('|')]
    feat_ids += [data_vocab['feats2id']['_']] * (max_num_feats - len(feat_ids))
    values = [sent_idx, token_idx, analysis_id, morpheme_id]
    values += [lattice_data_row.is_gold]
    values += [form_id, lemma_id, tag_id]
    values += feat_ids
    return values


def _to_token_row_values(lattice_data_row, data_vocab):
    sent_idx = lattice_data_row.sent_id
    token_idx = lattice_data_row.token_id
    analysis_idx = lattice_data_row.analysis_id
    morpheme_idx = lattice_data_row.morpheme_id
    form_id = data_vocab['form2id'][str(lattice_data_row.form)]
    lemma_id = data_vocab['lemma2id'][str(lattice_data_row.lemma)]
    tag_id = data_vocab['tag2id'][str(lattice_data_row.tag)]
    feats_id = data_vocab['feats_str2id'][str(lattice_data_row.feats)]
    # morpheme_id = ['pref', 'host', 'suff'].index(row.morpheme_type) if morpheme_type else row.morpheme_id
    values = [sent_idx, token_idx, analysis_idx, morpheme_idx]
    values += [form_id, lemma_id, tag_id, feats_id]
    return values


def _to_morpheme_row_values(lattice_data_row, sent_lengths, data_vocab):
    sent_idx = lattice_data_row.sent_id
    token_idx = lattice_data_row.token_id
    analysis_idx = lattice_data_row.analysis_id
    morpheme_idx = lattice_data_row.morpheme_id
    segment_idx = lattice_data_row.Index + 1 - sent_lengths[sent_idx]
    form_id = data_vocab['form2id'][str(lattice_data_row.form)]
    lemma_id = data_vocab['lemma2id'][str(lattice_data_row.lemma)]
    tag_id = data_vocab['tag2id'][str(lattice_data_row.tag)]
    feats_id = data_vocab['feats_str2id'][str(lattice_data_row.feats)]
    values = [sent_idx, token_idx, analysis_idx, morpheme_idx, segment_idx]
    values += [form_id, lemma_id, tag_id, feats_id]
    return values


# ldf - lattice data frame
def _get_seq_samples(lattices_df, data_vocab):
    token_char_ids = {}
    column_names = ['sent_idx', 'seq_idx', 'char_idx', 'seq_id', 'char_id']
    seq_row_values = [_to_tokens_row_values(lattice_data_row, data_vocab, token_char_ids)
                      for lattice_data_row in lattices_df.itertuples()]
    seq_samples_df = pd.DataFrame([seq_row for sent_seq_rows in seq_row_values
                                   for seq_row in sent_seq_rows], columns=column_names)

    # Input sequence samples
    num_samples = seq_samples_df.sent_idx.max()
    max_len = seq_samples_df.seq_idx.max()
    max_chars = seq_samples_df.char_idx.max()
    seq_samples = np.zeros((num_samples, max_len, max_chars, 2), dtype=np.int)
    sent_indices = seq_samples_df.sent_idx.values - 1
    seq_indices = seq_samples_df.seq_idx.values - 1
    char_indices = seq_samples_df.char_idx.values - 1
    values = seq_samples_df[['seq_id', 'char_id']]
    seq_samples[sent_indices, seq_indices, char_indices] = values
    # Input sequence and char lengths
    seq_length_samples = np.zeros((num_samples, max_len, 2), dtype=np.int)
    char_lengths = seq_samples_df.groupby(['sent_idx', 'seq_idx'])[['char_idx']].max().squeeze()
    sent_indices = [v[0] - 1 for v in char_lengths.index.values]
    seq_indices = [v[1] - 1 for v in char_lengths.index.values]
    seq_length_samples[sent_indices, seq_indices, 1] = char_lengths.values
    seq_lengths = seq_samples_df.groupby(['sent_idx'])[['seq_idx']].max().squeeze()
    sent_indices = [v - 1 for v in seq_lengths.index.values]
    seq_length_samples[sent_indices, 0, 0] = seq_lengths.values

    # num_sample (sent_idx.max()) may be greater than the actual number of samples if there are gaps in sent indices.
    # So we need to only keep the entries in the array that correspond to actual sentence indices.
    # Note - this technique is memory intensive. When I applied this technique in _get_lattice_analysis_samples the
    # memory consumption went up to 100GB. That is why I modified _get_lattice_analysis_samples to construct a single
    # array with num_sample set to sent_idx.unique().size and manually loop over the data and fill it in order to avoid
    # the slicing.
    return (seq_samples[seq_samples_df.sent_idx.unique() - 1],
            seq_length_samples[seq_samples_df.sent_idx.unique() - 1])


def _get_lattice_analysis_samples(lattice_df, data_vocab, max_morphemes, max_feats_len):
    indices_column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morpheme_column_names = ['is_gold', 'form_id', 'lemma_id', 'tag_id']
    feat_column_names = [f'feat{i+1}_id' for i in range(max_feats_len)]
    column_names = indices_column_names + morpheme_column_names + feat_column_names
    lattice_values = [_to_lattice_row_values(lattice_data_row, max_feats_len, data_vocab)
                      for lattice_data_row in lattice_df.itertuples()]
    lattice_samples_df = pd.DataFrame(lattice_values, columns=column_names)

    # Morpheme samples
    # num_samples = lattice_samples_df.sent_idx.max()
    num_samples = lattice_samples_df.sent_idx.unique().size
    max_len = lattice_samples_df.token_idx.max()
    max_analyses = lattice_samples_df.analysis_idx.max() + 1
    morpheme_len = len(morpheme_column_names) + len(feat_column_names)
    samples_shape = (num_samples, max_len, max_analyses, max_morphemes, morpheme_len)

    # https://stackoverflow.com/questions/54615882/how-to-convert-a-pandas-multiindex-dataframe-into-a-3d-array
    # http://xarray.pydata.org/en/stable/
    lattice_analysis_samples = np.zeros(samples_shape, dtype=np.int)
    samples_df = lattice_samples_df.set_index(['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx'])
    sid = -1
    row_sid = -1
    for row in sorted(samples_df.itertuples()):
        if row[0][0] - 1 > row_sid:
            row_sid = row[0][0] - 1
            sid += 1
        tid = row[0][1] - 1
        aid = row[0][2]
        mid = row[0][3]
        lattice_analysis_samples[(sid, tid, aid, mid)] = row[1:]
    length_sampels_shape = (num_samples, max_len)
    lattice_analysis_length_samples = np.zeros(length_sampels_shape, dtype=np.int)
    analysis_lengths = lattice_samples_df.groupby(['sent_idx', 'token_idx'])['analysis_idx'].max()
    sid = -1
    row_sid = -1
    for row in sorted(analysis_lengths.iteritems()):
        if row[0][0] - 1 > row_sid:
            row_sid = row[0][0] - 1
            sid += 1
        tid = row[0][1] - 1
        lattice_analysis_length_samples[(sid, tid)] = row[1] + 1
    return lattice_analysis_samples, lattice_analysis_length_samples


def _get_fixed_analysis_samples(analyses_df, data_vocab, max_morphemes):
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id', 'feats_id']
    column_names += morph_column_names
    morpheme_values = [_to_token_row_values(data_row, data_vocab) for data_row in analyses_df.itertuples()]
    analysis_samples_df = pd.DataFrame(morpheme_values, columns=column_names)

    # Analysis samples
    num_samples = analysis_samples_df.sent_idx.max()
    max_len = analysis_samples_df.token_idx.max()
    default_morph_values = [data_vocab['form2id']['_'], data_vocab['lemma2id']['_'], data_vocab['tag2id']['_'],
                            data_vocab['feats2id']['_']]
    analysis_samples = np.array(default_morph_values, dtype=np.int)
    analysis_samples = np.tile(analysis_samples, (num_samples, max_len, max_morphemes, 1))
    sent_indices = analysis_samples_df['sent_idx'].values - 1
    token_indices = analysis_samples_df['token_idx'].values - 1
    morpheme_indices = analysis_samples_df['morpheme_idx'].values
    values = analysis_samples_df[morph_column_names].values
    analysis_samples[sent_indices, token_indices, morpheme_indices] = values

    # Set <PAD>
    # Find sentence boundary indices - this is used to get the number of tokens in each sentence
    token_mask = [bool(sent_indices[i] != sent_indices[i + 1]) for i in range(len(sent_indices) - 1)] + [True]

    # Use sentence boundary indices as start position for filling token indices
    fill_token_indices = [ii for i in token_indices[token_mask] for ii in range(i + 1, max_len)]

    # Now construct the sentence indices corresponding to the token indices
    fill_sent_indices = analysis_samples_df.sent_idx.unique() - 1
    fill_sent_indices = [fill_sent_indices[j].item() for j, i in enumerate(token_indices[token_mask])
                         for ii in range(i + 1, max_len)]
    analysis_samples[fill_sent_indices, fill_token_indices] = 0

    # num_sample (sent_idx.max()) may be greater than the actual number of samples if there are gaps in sent indices.
    # So we need to only keep the entries in the array that correspond to actual sentence indices.
    return analysis_samples[analysis_samples_df.sent_idx.unique() - 1]


# Variable sized analyses (with special <EOT> morpheme)
def _get_var_morpheme_samples(analyses_df, data_vocab, max_morphemes):
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id', 'feats_id']
    column_names += morph_column_names
    morpheme_values = [_to_token_row_values(data_row, data_vocab) for data_row in analyses_df.itertuples()]
    morpheme_samples_df = pd.DataFrame(morpheme_values, columns=column_names)

    # Morpheme samples
    max_sample = morpheme_samples_df.sent_idx.max()
    max_len = morpheme_samples_df.token_idx.max()
    morpheme_samples = np.zeros((max_sample, max_len, max_morphemes + 1, len(morph_column_names)), dtype=np.int)
    sent_indices = morpheme_samples_df['sent_idx'].values - 1
    token_indices = morpheme_samples_df['token_idx'].values - 1
    morpheme_indices = morpheme_samples_df['morpheme_idx'].values
    values = morpheme_samples_df[morph_column_names].values
    morpheme_samples[sent_indices, token_indices, morpheme_indices] = values

    # Set <EOT>
    # Find sentence boundary indices - this is used to get the number of tokens in each sentence
    token_mask = [bool(sent_indices[i] != sent_indices[i + 1]) for i in range(len(sent_indices) - 1)] + [True]

    # Use sentence boundary indices as start position for filling token indices
    fill_token_indices = [ii for i in token_indices[token_mask] for ii in range(i + 1, max_len)]

    # (max_sample > # of samples) since some ZVL samples were filtered, so you have to map to the correct sentence id
    fill_sent_indices = morpheme_samples_df.sent_idx.unique() - 1

    # Now construct the sentence indices corresponding to the token indices
    fill_sent_indices = [fill_sent_indices[j].item() for j, i in enumerate(token_indices[token_mask]) for ii in
                         range(i + 1, max_len)]

    # Fill Values
    morpheme_samples[fill_sent_indices, fill_token_indices] = 0

    # Find token boundary indices - this is used to get number of morphemes in each token analysis
    token_mask = [bool(token_indices[i] != token_indices[i + 1]) for i in range(len(token_indices) - 1)] + [True]

    # Use token boundary indices to get the <EOT> morpheme indices (which are zero based), token indices and sentence
    # indices (which are 1 based)
    fill_morpheme_indices = morpheme_indices[token_mask] + 1
    fill_token_indices = token_indices[token_mask]
    fill_sent_indices = sent_indices[token_mask]

    # Fill values
    eot_values = [data_vocab['form2id']['<EOT>'], data_vocab['lemma2id']['<EOT>'], data_vocab['tag2id']['<EOT>'],
                  data_vocab['feats2id']['<EOT>']]
    morpheme_samples[fill_sent_indices, fill_token_indices, fill_morpheme_indices] = eot_values

    # num_sample (sent_idx.max()) may be greater than the actual number of samples if there are gaps in sent indices.
    # So we need to only keep the entries in the array that correspond to actual sentence indices.
    return morpheme_samples[morpheme_samples_df.sent_idx.unique() - 1]


def _load_data(root_path, partition, ner_feat, ner_only, baseline, data_type=None):
    dataset = {}
    ner_suff = f'ner_pos_{ner_feat}' if not ner_only else f'ner_pos_{ner_feat}_only'
    for partition_type in partition:
        if data_type:
            file_path = root_path / f'{partition_type}-{baseline}-{data_type}.{ner_suff}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}-{baseline}.{ner_suff}.lattices.csv'

        # Bug fix: load the actual tokens 'NA', 'nan', etc. (this actually happens in the tr_imst treebank)
        dataset[partition_type] = pd.read_csv(str(file_path), keep_default_na=False)
        print(f'{file_path.name} data size: {len(dataset[partition_type])}')
    return dataset


def _remove_infused_analyses(lattices_dataset):
    uninfused_dataset = {}
    for partition_type in lattices_dataset:
        df = lattices_dataset[partition_type]
        uninfused_dataset[partition_type] = df[df.is_inf != True]
        print(f'{partition_type} uninfused data size: {len(uninfused_dataset[partition_type])}')
    return uninfused_dataset


def _load_lattices_data_samples(infused_lattices_dataset, uninfused_lattices_dataset, data_vocab):
    token_samples = {t: _get_seq_samples(infused_lattices_dataset[t], data_vocab) for t in infused_lattices_dataset}

    # All variable sized attributes such as the number of morpheme per analysis or number of features per morpheme
    # must be the same across all partitions (train, dev, test) so all partition arrays are the same fixed size.
    # max_morphemes = {t: lattices_dataset[t].morpheme_id.max() + 1 for t in partition}
    # max_feats_len = {t: max(_get_feats_len(lattices_dataset[t].feats.values)) for t in partition}
    max_morphemes = max([infused_lattices_dataset[t].morpheme_id.max() + 1 for t in infused_lattices_dataset])
    max_feats_len = max([max(_get_feats_len(infused_lattices_dataset[t].feats.values)) for t in infused_lattices_dataset])

    infused_morph_samples = {t: _get_lattice_analysis_samples(infused_lattices_dataset[t], data_vocab, max_morphemes, max_feats_len) for t in infused_lattices_dataset}
    uninfused_morph_samples = {t: _get_lattice_analysis_samples(uninfused_lattices_dataset[t], data_vocab, max_morphemes, max_feats_len) for t in uninfused_lattices_dataset}
    return token_samples, infused_morph_samples, uninfused_morph_samples, data_vocab


def _spmrl_feats_to_str(feats):
    feat_keys = ['gen', 'num', 'per', 'tense', 'suf_gen', 'suf_num', 'suf_per']
    feat_str_rows = []
    for token_feats in feats:
        token_feat_str_rows = []
        for morpheme_feats in token_feats:
            morpheme_feats = np.unique(morpheme_feats)
            if morpheme_feats.size == 1:
                token_feat_str_rows.append(morpheme_feats.item())
                continue
            morpheme_feats_dict = {f[0]: f[1] for f in [f.split("=") for f in morpheme_feats[morpheme_feats != '_']]}
            s = []
            for feat_name in feat_keys:
                if feat_name in morpheme_feats_dict:
                    if feat_name == 'tense':
                        feat_value = morpheme_feats_dict[feat_name]
                        s.append(f'{feat_name}={feat_value}')
                    else:
                        for feat_value in morpheme_feats_dict[feat_name]:
                            s.append(f'{feat_name}={feat_value}')
            token_feat_str_rows.append('|'.join(s))
        feat_str_rows.append(np.array(token_feat_str_rows))
    return np.stack(feat_str_rows)


def _ud_feats_to_str(feats):
    feat_str_rows = []
    for token_feats in feats:
        token_feat_str_rows = []
        for morpheme_feats in token_feats:
            morpheme_feats = np.unique(morpheme_feats)
            if morpheme_feats.size == 1:
                token_feat_str_rows.append(morpheme_feats.item())
                continue
            morpheme_feats_dict = defaultdict(list)
            for f in [f.split("=") for f in morpheme_feats[morpheme_feats != '_']]:
                morpheme_feats_dict[f[0]].append(f[1])
            s = [f'{name}={value}' for name in morpheme_feats_dict for value in morpheme_feats_dict[name]]
            token_feat_str_rows.append('|'.join(s))
        feat_str_rows.append(np.array(token_feat_str_rows))
    return np.stack(feat_str_rows)


# API ##################################################################################################################
def get_num_token_tags(multi_tag_ids, data_vocab):
    multi_tags = _to_tag_vec(multi_tag_ids, data_vocab)
    return _get_multi_tags_len(multi_tags).sum(axis=2).max()


def tag_ids_to_tags(tag_ids, data_vocab):
    return _to_tag_vec(tag_ids, data_vocab)


def tags_to_tag_ids(tags, data_vocab):
    return _to_tag_id_vec(tags, data_vocab)


def tag_ids_to_spmrl_lattice(tag_ids, token_mask, data_vocab):
    return _tag_ids_to_token_lattice(tag_ids, token_mask, data_vocab, _spmrl_feats_to_str)


def lattice_ids_to_spmrl_lattice(token_lattice_ids, data_vocab):
    return _lattice_ids_to_token_lattice(token_lattice_ids, data_vocab, _spmrl_feats_to_str)


def tag_ids_to_ud_lattice(tag_ids, token_mask, data_vocab):
    return _tag_ids_to_token_lattice(tag_ids, token_mask, data_vocab, _ud_feats_to_str)


def lattice_ids_to_ud_lattice(token_lattice_ids, data_vocab):
    return _lattice_ids_to_token_lattice(token_lattice_ids, data_vocab, _ud_feats_to_str)


def token_ids_to_tokens(token_ids, token_mask, vocab):
    tokens = token_ids[:, :, 0, 0][token_mask]
    return _to_token_vec(tokens, vocab)


def eval_samples(samples):
    gold_df = _concat_data([_to_data(sample[0], sample[1]) for sample in samples])
    pred_df = _concat_data([_to_data(sample[0], sample[2]) for sample in samples])
    return _tag_eval(gold_df, pred_df)


def seg_eval_samples(samples):
    gold_df = _concat_data([_to_data(sample[0], sample[1]) for sample in samples])
    pred_df = _concat_data([_to_data(sample[0], sample[2]) for sample in samples])
    return _seg_tag_eval(gold_df, pred_df)


def load_vocab(root_path, ner_feat, ner_only, baseline, la_name, tb_name, seq_type='', ma_name=None):
    if seq_type == 'lattice':
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'{seq_type}' / ma_name / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    elif seq_type.endswith('-mtag'):
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / 'seq' / f'{seq_type}' / f'vocab-{baseline}'
    else:
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    return _load_vocab_entries(vocab_dir_path)


def load_ft_emb(root_path, ft_root_path, ner_feat, ner_only, baseline, data_vocab, la_name, tb_name, seq_type='', ma_name=None):
    ft.ft_model = None
    if seq_type == 'lattice':
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'{seq_type}' / ma_name / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    elif seq_type.endswith('-mtag'):
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / 'seq' / f'{seq_type}' / f'vocab-{baseline}'
    else:
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    return _load_morpheme_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def load_data_samples(root_path, partition, ner_feat, ner_only, baseline, la_name, tb_name, seq_type='', ma_name=None):
    if seq_type == 'lattice':
        data_dir = root_path / la_name / f'{tb_name}-NER-POS' / f'{seq_type}' / ma_name
        infused_lattices_dataset = _load_data(data_dir, partition, ner_feat, ner_only, baseline, 'inf')
        uninfused_lattices_dataset = _remove_infused_analyses(infused_lattices_dataset)
        data_vocab = load_vocab(root_path, ner_feat, ner_only, baseline, la_name, tb_name, seq_type, ma_name=ma_name)
        seq_samples, infused_morph_samples, uninfused_morph_samples, data_vocab = _load_lattices_data_samples(infused_lattices_dataset, uninfused_lattices_dataset, data_vocab)
        return seq_samples, infused_morph_samples, uninfused_morph_samples, data_vocab
    if seq_type.endswith('-mtag'):
        data_dir = root_path / la_name / tb_name / 'seq' / f'{seq_type}'
        base_dataset = _load_data(data_dir, partition, baseline, 'mtag')
        data_vocab = load_vocab(root_path, baseline, la_name, tb_name, seq_type)
        max_morphemes = max([base_dataset[t].morpheme_id.max() + 1 for t in base_dataset])
        morph_samples = {t: _get_fixed_analysis_samples(base_dataset[t], data_vocab, max_morphemes) for t in base_dataset}
        seq_samples = {t: _get_seq_samples(base_dataset[t], data_vocab) for t in base_dataset}
        return seq_samples, morph_samples, data_vocab
    data_dir = root_path / la_name / f'{tb_name}-NER-POS'
    base_dataset = _load_data(data_dir, partition, ner_feat, ner_only, baseline)
    data_vocab = load_vocab(root_path, ner_feat, ner_only, baseline, la_name, tb_name, seq_type)
    seq_samples = {t: _get_seq_samples(base_dataset[t], data_vocab) for t in base_dataset}
    if baseline == 'gold':
        max_morphemes = max([base_dataset[t].morpheme_id.max() + 1 for t in base_dataset])
        morph_samples = {t: _get_var_morpheme_samples(base_dataset[t], data_vocab, max_morphemes) for t in base_dataset}
        return seq_samples, morph_samples, data_vocab
    gold_dataset = _load_data(data_dir, partition, 'gold')
    gold_max_morphemes = max([gold_dataset[t].morpheme_id.max() + 1 for t in gold_dataset])
    base_max_morphemes = max([base_dataset[t].morpheme_id.max() + 1 for t in base_dataset])
    max_morphemes = max([gold_max_morphemes, base_max_morphemes])
    gold_morph_samples = {t: _get_var_morpheme_samples(gold_dataset[t], data_vocab, max_morphemes) for t in gold_dataset}
    base_morph_samples = {t: _get_var_morpheme_samples(base_dataset[t], data_vocab, max_morphemes) for t in base_dataset}
    if la_name == 'en':
        train_seq = seq_samples['train']
        train_gold_morph_samples = gold_morph_samples['train']
        train_base_morph_samples = base_morph_samples['train']
        misaligned_ids_file_path = data_dir / 'train-udpipe-misaligned-sent-ids.txt'
        t = misaligned_ids_file_path.read_text()
        remove_indices = [int(line) - 1 for line in t.split()]
        seq_samples['train'] = np.delete(train_seq[0], remove_indices, axis=0), np.delete(train_seq[1], remove_indices, axis=0)
        gold_morph_samples['train'] = np.delete(train_gold_morph_samples, remove_indices, axis=0)
        base_morph_samples['train'] = np.delete(train_base_morph_samples, remove_indices, axis=0)
    return seq_samples, gold_morph_samples, base_morph_samples, data_vocab


def to_conllu_mono_lattice_str(tokens, analyses):
    conllu_column_names = ['token_id', 'form', 'lemma', 'cpostag', 'upostag', 'feats', 'head', 'deprel', 'deps', 'misc']
    rows = []
    morph_id = 1
    for token, analysis in zip(tokens, [analysis[analysis != '<PAD>'].reshape(analysis.shape[0], -1) for analysis in analyses]):
        # Duplicate the postag value (UPOSTAG, XPOSTAG)
        analysis = np.repeat(analysis, repeats=[1, 1, 2, 1], axis=0)
        morphemes = analysis.astype('object').T
        # Add token (multi-word) line: x-y token _ _ _ _ _ _ _ _
        if len(morphemes) > 1:
            rows.append([f'{morph_id}-{morph_id + len(morphemes) - 1}', token] + ['_'] * 8)
        for morpheme in morphemes:
            if len(morphemes) == 1:
                morpheme[0] = token
            rows.append([morph_id] + morpheme.tolist() + [morph_id - 1] + ['_'] * 3)
            morph_id += 1
    df = pd.DataFrame(rows, columns=conllu_column_names)
    return df.to_csv(header=False, index=False, sep='\t', escapechar=None, quoting=csv.QUOTE_NONE)


def to_lattice_sample(tokens, lattice_ids, gold_indices, data_vocab, lattice_id_to_lattice_func):
    lattice_gold_indices = gold_indices.reshape(-1)[:, None].repeat(lattice_ids.shape[1], 1).transpose()
    # lattice_is_inf = is_inf.repeat(7).repeat(4).reshape(is_inf.shape[1], is_inf.shape[2], 7, 4)
    lattice_tokens = tokens[:, None].repeat(lattice_ids.shape[1], 1).transpose()
    lattice_analyses = []
    for i in range(lattice_ids.shape[1]):
        analyses = lattice_id_to_lattice_func(lattice_ids[:, i], data_vocab)
        lattice_analyses.append(analyses)
    lattice_analyses = np.stack(lattice_analyses, axis=1).transpose((0, 1, 3, 2))
    lattice_analyses_mask = (lattice_analyses != '<PAD>').nonzero()
    lattice_analyses = lattice_analyses[lattice_analyses_mask[0], lattice_analyses_mask[1],
                                        lattice_analyses_mask[2], lattice_analyses_mask[3]]
    lattice_analyses = lattice_analyses.reshape(-1, 4)
    lattice_analyses_mask = [mask.reshape(-1, 4)[:, 0] for mask in lattice_analyses_mask]

    lattice_analyses_from_indices = np.zeros((lattice_analyses.shape[0], 1), dtype=np.int)
    lattice_analyses_to_indices = np.ones((lattice_analyses.shape[0], 1), dtype=np.int)
    lattice_analyses_token_indices = lattice_analyses_mask[0] + 1
    lattice_analyses_analysis_indices = lattice_analyses_mask[1]
    lattice_analyses_morpheme_indices = lattice_analyses_mask[2]
    lattice_analyses_gold_indices = lattice_gold_indices[lattice_analyses_mask[1], lattice_analyses_mask[0]]
    lattice_analyses_is_gold = lattice_analyses_analysis_indices == lattice_analyses_gold_indices
    lattice_analyses_tokens = lattice_tokens[lattice_analyses_mask[1], lattice_analyses_mask[0]]

    lattice = np.concatenate([lattice_analyses_from_indices, lattice_analyses_to_indices, lattice_analyses,
                              lattice_analyses_token_indices[:, None], lattice_analyses_tokens[:, None],
                              lattice_analyses_is_gold[:, None], lattice_analyses_analysis_indices[:, None],
                              lattice_analyses_morpheme_indices[:, None]],
                             axis=1)
    return lattice


def save_as_conllu(samples, out_file_path):
    with open(str(out_file_path), 'w') as f:
        for sample in samples:
            lattice_str = to_conllu_mono_lattice_str(sample[0], sample[-1])
            f.write(lattice_str)
            f.write('\n')


def save_as_lattice_samples(lattices, out_file_path):
    lattices = [np.concatenate([np.full((lattice.shape[0], 1), i + 1, dtype=np.int), lattice], axis=1) for i, lattice in
                enumerate(lattices)]
    df = pd.DataFrame(np.concatenate(lattices))
    df.to_csv(out_file_path)
# API ##################################################################################################################


def _save_vocab(root_path, partition, ner_feat, ner_only, baseline, la_name, tb_name, seq_type='', ma_name=None):
    if seq_type == 'lattice':
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'{seq_type}' / ma_name / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
        lattices_dataset, base_dataset = tb.tb_load_lattices(root_path, partition, ner_feat, ner_only, baseline, la_name, tb_name, ma_name, 'inf')
        lattices_vocab = _get_vocab(lattices_dataset)
        base_vocab = _get_vocab(base_dataset)
        data_vocab = _get_vocabs_union(lattices_vocab, base_vocab)
    else:
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
        base_dataset = tb.tb_load_base(root_path, partition, ner_feat, ner_only, baseline, la_name, tb_name)
        if baseline != 'gold':
            gold_dataset = tb.tb_load_base(root_path, partition, ner_feat, ner_only, 'gold', la_name, tb_name)
            gold_vocab = _get_vocab(gold_dataset)
            base_vocab = _get_vocab(base_dataset)
            data_vocab = _get_vocabs_union(gold_vocab, base_vocab)
        else:
            data_vocab = _get_vocab(base_dataset)
    _save_vocab_files(vocab_dir_path, data_vocab)


def _save_ft_emb(root_path, ft_root_path, ner_feat, ner_only, baseline, la_name, tb_name, seq_type='', ma_name=None):
    if seq_type == 'lattice':
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'{seq_type}' / ma_name / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    elif seq_type:
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / 'seq' / f'{seq_type}' / f'vocab-{baseline}'
    else:
        vocab_dir_path = root_path / la_name / f'{tb_name}-NER-POS' / f'vocab-{baseline}' / (f'{ner_feat}' if not ner_only else f'{ner_feat}_only')
    data_vocab = _load_vocab_entries(vocab_dir_path)
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    ft.ft_model = None
    _save_morpheme_ft_emb_files(vocab_dir_path, ft_model_path, data_vocab)
     # _save_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def main():
    f = 1
    o = False
    # p = 0
    scheme = 'UD'
    partition = ['dev', 'test', 'train']
    ner_feat = ['plo', 'nocat', 'full']
    root_path = Path.home() / f'dev/aseker00/modi/tb/{scheme}'
    ft_path = Path.home() / 'dev/fastText'
    tb_name = 'HTB'
    ma_name = 'heblex'
    la_name = 'he'
    _save_vocab(root_path, partition, ner_feat[f], o, 'gold', la_name, tb_name)
    _save_ft_emb(root_path, ft_path, ner_feat[f], o, 'gold', la_name, tb_name)
    _save_vocab(root_path, partition, ner_feat[f], o, 'gold', la_name, tb_name, 'lattice', ma_name)
    _save_ft_emb(root_path, ft_path, ner_feat[f], o, 'gold', la_name, tb_name, 'lattice', ma_name)

    # token_samples, morph_samples, data_vocab = load_data_samples(root_path, partition, ner_feat[f], o, 'gold', la_name, tb_name)
    # for partition_type in partition:
    #     print(f'{token_samples[partition_type][0].shape} {partition_type} gold token samples, '
    #           f'{morph_samples[partition_type].shape} {partition_type} gold morpheme samples')
    # token_samples, infused_morph_samples, uninfused_morph_samples, data_vocab = load_data_samples(root_path, partition, ner_feat[f], o, 'gold', la_name, tb_name, 'lattice', ma_name)
    # for partition_type in partition:
    #     print(f'{token_samples[partition_type][0].shape} {partition_type} gold token samples, '
    #           f'{infused_morph_samples[partition_type][0].shape} {partition_type} infused morpheme samples, '
    #           f'{uninfused_morph_samples[partition_type][0].shape} {partition_type} uninfused morpheme samples')


if __name__ == '__main__':
    main()
