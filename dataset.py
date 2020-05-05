from collections import defaultdict, Counter
import fasttext_emb as ft
import pandas as pd
import numpy as np


def get_vocab(dataset):
    tokens, forms, lemmas, tags, feats = set(), set(), set(), set(), set()
    for partition_type in dataset:
        for df in dataset[partition_type]:
            tokens.update(set(df.token))
            forms.update(set(df.form))
            lemmas.update(set(df.lemma))
            tags.update(set(df.tag))
            feats.update(set(df.feats))
    chars = set([c for t in tokens for c in list(t)])
    return to_vocab(tokens, chars, forms, lemmas, tags, feats)


def to_vocab(tokens, chars, forms, lemmas, tags, feats):
    tokens = ['<PAD>'] + sorted(tokens)
    chars = ['<PAD>'] + sorted(list(chars))
    for m in [forms, lemmas, tags, feats]:
        if '_' in m:
            m.remove('_')
    forms = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(forms))
    lemmas = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(lemmas))
    tags = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(tags))
    feats_str = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(feats))
    feat_set = set()
    for f in feats:
        fmap = defaultdict(list)
        for ff in f.split('|'):
            name, value = ff.split('=')
            fmap[name].append(value)
        for name in fmap:
            value = ''.join(fmap[name])
            feat_set.add(f'{name}={value}')
    feats = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(feat_set)
    # feats = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(set([ff for f in feats for ff in f.split('|')]))
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


def save_vocab(root_path, vocab):
    for vocab_type in ['tokens', 'chars', 'forms', 'lemmas', 'tags', 'feats', 'feats_str']:
        vocab_file_path = root_path / f'{vocab_type}.txt'
        with open(str(vocab_file_path), 'w') as f:
            f.write('\n'.join(vocab[vocab_type]))
        print(f'{vocab_type} vocab size: {len(vocab[vocab_type])}')


def load_vocab(root_path):
    vocab = {}
    vocab_types = {'tokens': 'token2id', 'chars': 'char2id', 'forms': 'form2id', 'lemmas': 'lemma2id', 'tags': 'tag2id',
                   'feats': 'feats2id', 'feats_str': 'feats_str2id'}
    for vocab_type in vocab_types:
        vocab_file_path = root_path / f'{vocab_type}.txt'
        with open(str(vocab_file_path)) as f:
            entries = [line.strip() for line in f.readlines()]
            entry2ids = {v: k for k, v in enumerate(entries)}
            vocab[vocab_type] = entries
            vocab[vocab_types[vocab_type]] = entry2ids
        print(f'{vocab_type} vocab size: {len(vocab[vocab_type])}')
    return vocab


def save_ft_vec(root_path, ft_root_path):
    vocab = load_vocab(root_path)
    ft_model_path = ft_root_path / 'models/cc.he.300.bin'
    chars_vec_file_path = root_path / 'chars.vec'
    tokens_vec_file_path = root_path / 'tokens.vec'
    forms_vec_file_path = root_path / 'forms.vec'
    lemmas_vec_file_path = root_path / 'lemmas.vec'
    if chars_vec_file_path.exists():
        chars_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, vocab['chars'])
    if tokens_vec_file_path.exists():
        tokens_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, vocab['tokens'])
    if forms_vec_file_path.exists():
        forms_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, forms_vec_file_path, vocab['forms'])
    if lemmas_vec_file_path.exists():
        lemmas_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, lemmas_vec_file_path, vocab['lemmas'])


def load_ft_vec(root_path, ft_root_path, vocab):
    ft_model_path = ft_root_path / 'models/cc.he.300.bin'
    chars_vec_file_path = root_path / 'chars.vec'
    tokens_vec_file_path = root_path / 'tokens.vec'
    forms_vec_file_path = root_path / 'forms.vec'
    lemmas_vec_file_path = root_path / 'lemmas.vec'
    return (ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, vocab['chars']),
            ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, vocab['tokens']),
            ft.load_embedding_weight_matrix(ft_model_path, forms_vec_file_path, vocab['forms']),
            ft.load_embedding_weight_matrix(ft_model_path, lemmas_vec_file_path, vocab['lemmas']))


def to_tokens_row(row, vocab, token_char_ids):
    if row.token in token_char_ids:
        token_id, char_ids = token_char_ids[row.token]
    else:
        token_id = vocab['token2id'][row.token]
        char_ids = [vocab['char2id'][c] for c in row.token]
        token_char_ids[row.token] = (token_id, char_ids)
    return [[row.sent_id, row.token_id, i + 1, token_id, char_id] for i, char_id in enumerate(char_ids)]


def get_tokens_arr(df, vocab, column_names):
    token_char_ids = {}
    sample_rows = [to_tokens_row(row, vocab, token_char_ids) for row in df.itertuples()]
    samples_df = pd.DataFrame([row for sample in sample_rows for row in sample], columns=column_names)
    num_samples = samples_df.sent_idx.max()
    max_len = samples_df.token_idx.max()
    max_chars = samples_df.char_idx.max()
    # Samples
    samples_arr = np.zeros((num_samples, max_len, max_chars, 2), dtype=np.int)
    sent_indices = samples_df.sent_idx.values - 1
    token_indices = samples_df.token_idx.values - 1
    char_indices = samples_df.char_idx.values - 1
    values = samples_df[['token_id', 'char_id']].values
    samples_arr[sent_indices, token_indices, char_indices] = values
    # Token and char lengths
    length_arr = np.zeros((num_samples, max_len, 2), dtype=np.int)
    char_lengths = samples_df.groupby(['sent_idx', 'token_idx'])[['char_idx']].max().squeeze()
    sent_indices = [v[0] - 1 for v in char_lengths.index.values]
    token_indices = [v[1] - 1 for v in char_lengths.index.values]
    length_arr[sent_indices, token_indices, 1] = char_lengths.values
    token_lengths = samples_df.groupby(['sent_idx'])[['token_idx']].max().squeeze()
    sent_indices = [v - 1 for v in token_lengths.index.values]
    length_arr[sent_indices, 0, 0] = token_lengths.values
    return samples_arr[samples_df.sent_idx.unique() - 1], length_arr[samples_df.sent_idx.unique() - 1]


def load_data_samples(root_path, partition, tag_type, morph_seq_func):
    vocab = load_vocab(root_path / 'vocab')
    dataset = {}
    max_morphemes = {}
    token_column_names = ['sent_idx', 'token_idx', 'char_idx', 'token_id', 'char_id']
    for partition_type in partition:
        print(f'loading {partition_type}-{tag_type} samples')
        file_path = root_path / f'{partition_type}-{tag_type}.csv'
        dataset[partition_type] = pd.read_csv(str(file_path), index_col=0)
        max_morphemes[partition_type] = dataset[partition_type].morpheme_id.max() + 1
    token_samples = {t: get_tokens_arr(dataset[t], vocab, token_column_names) for t in dataset}
    morph_samples = {t: morph_seq_func(dataset[t], vocab, max_morphemes[partition[-1]]) for t in dataset}
    return token_samples, morph_samples, vocab


def feats_to_str(feats):
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


to_form_vec = np.vectorize(lambda x, vocab: vocab['forms'][x])
to_lemma_vec = np.vectorize(lambda x, vocab: vocab['lemmas'][x])
to_tag_vec = np.vectorize(lambda x, vocab: vocab['tags'][x])
to_feat_vec = np.vectorize(lambda x, vocab: vocab['feats'][x])
to_token_vec = np.vectorize(lambda x, vocab: vocab['tokens'][x])
to_form_id_vec = np.vectorize(lambda x, vocab: vocab['form2id'][x])
to_lemma_id_vec = np.vectorize(lambda x, vocab: vocab['lemma2id'][x])
to_tag_id_vec = np.vectorize(lambda x, vocab: vocab['tag2id'][x])
to_feat_id_vec = np.vectorize(lambda x, vocab: vocab['feat2id'][x])
to_token_id_vec = np.vectorize(lambda x, vocab: vocab['token2id'][x])
get_multi_tags_len = np.vectorize(lambda x: len(x.split('-')))
get_feats_len = np.vectorize(lambda x: len(x.split('|')))


def tag_eval(gold_df, pred_df):
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


def seg_tag_eval(gold_df, pred_df):
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


def to_dataset(lattices):
    for i, df in enumerate(lattices):
        df.insert(0, 'sent_id', i + 1)
    return pd.concat(lattices)


def to_lattice_data(tokens, lattices):
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


def to_tokens(token_ids, token_mask, vocab):
    tokens = token_ids[:, :, 0, 0][token_mask]
    return to_token_vec(tokens, vocab)


def to_token_lattice(tag_ids, token_mask, vocab):
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
    token_forms = to_form_vec(token_form_ids, vocab)
    token_lemmas = to_lemma_vec(token_lemma_ids, vocab)
    token_tags = to_tag_vec(token_tag_ids, vocab)
    token_feats_str = feats_to_str(to_feat_vec(token_feat_ids, vocab))
    return np.stack([token_forms, token_lemmas, token_tags, token_feats_str], axis=1)


def to_tags_arr(lattice_df):
    values = [x[1].tag.values for x in lattice_df.groupby('token_id')]
    max_len = max([len(a) for a in values])
    tags_arr = np.full_like(lattice_df.tag.values, '<PAD>', shape=(len(values), max_len))
    for i, a in enumerate(values):
        tags_arr[i, :len(a)] = a
    return tags_arr


def eval_samples(samples):
    gold_df = to_dataset([to_lattice_data(sample[0], sample[1]) for sample in samples])
    pred_df = to_dataset([to_lattice_data(sample[0], sample[2]) for sample in samples])
    return tag_eval(gold_df, pred_df)


def seg_eval_samples(samples):
    gold_df = to_dataset([to_lattice_data(sample[0], sample[1]) for sample in samples])
    pred_df = to_dataset([to_lattice_data(sample[0], sample[2]) for sample in samples])
    return seg_tag_eval(gold_df, pred_df)
