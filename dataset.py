from collections import Counter
import fasttext_emb as ft
from treebank import *

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
    tags = np.full_like(lattice_df.tag.values, '<PAD>', shape=(len(values), max_len))
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
            for fff in ff.split('-'):
                if fff == '_':
                    continue
                name, value = fff.split('=')
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
    chars = set([c for t in tokens for c in list(t)])
    return _to_vocab(tokens, chars, forms, lemmas, tags, feats)


def _get_vocabs_union(dv1, dv2):
    tokens = set(dv1['tokens'] + dv2['tokens']).difference({'<PAD>', '<SOS>', '<EOT>'})
    forms = set(dv1['forms'] + dv2['forms']).difference({'<PAD>', '<SOS>', '<EOT>'})
    lemmas = set(dv1['lemmas'] + dv2['lemmas']).difference({'<PAD>', '<SOS>', '<EOT>'})
    tags = set(dv1['tags'] + dv2['tags']).difference({'<PAD>', '<SOS>', '<EOT>'})
    feats = set(dv1['feats'] + dv2['feats']).difference({'<PAD>', '<SOS>', '<EOT>'})
    chars = set([c for t in tokens for c in list(t)])
    return _to_vocab(tokens, chars, forms, lemmas, tags, feats)


def _save_vocab(data_vocab_dir_path, data_vocab):
    os.makedirs(data_vocab_dir_path, exist_ok=True)
    for key in ['tokens', 'chars', 'forms', 'lemmas', 'tags', 'feats', 'feats_str']:
        data_vocab_file_path = data_vocab_dir_path / f'{key}.txt'
        with open(str(data_vocab_file_path), 'w') as f:
            f.write('\n'.join(data_vocab[key]))
        print(f'{key} vocab size: {len(data_vocab[key])}')


def _load_vocab(data_vocab_dir_path):
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


def _save_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab):
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    if chars_vec_file_path.exists():
        chars_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, data_vocab['chars'])
    if tokens_vec_file_path.exists():
        tokens_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, data_vocab['tokens'])


def _save_morpheme_ft_emb(vocab_dir_path, ft_model_path, data_vocab):
    _save_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)
    forms_vec_file_path = vocab_dir_path / 'forms.vec'
    lemmas_vec_file_path = vocab_dir_path / 'lemmas.vec'
    if forms_vec_file_path.exists():
        forms_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, forms_vec_file_path, data_vocab['forms'])
    if lemmas_vec_file_path.exists():
        lemmas_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, lemmas_vec_file_path, data_vocab['lemmas'])


def _to_tokens_row_values(lattice_data_row, data_vocab, token_char_ids):
    if lattice_data_row.token in token_char_ids:
        token_id, char_ids = token_char_ids[str(lattice_data_row.token)]
    else:
        token_id = data_vocab['token2id'][str(lattice_data_row.token)]
        char_ids = [data_vocab['char2id'][c] for c in str(lattice_data_row.token)]
        token_char_ids[str(lattice_data_row.token)] = (token_id, char_ids)
    return [[lattice_data_row.sent_id, lattice_data_row.token_id, i + 1, token_id, char_id]
            for i, char_id in enumerate(char_ids)]


def _to_lattice_row_values(lattice_data_row, max_num_feats, data_vocab, infuse):
    form_id = data_vocab['form2id'][str(lattice_data_row.form)]
    lemma_id = data_vocab['lemma2id'][str(lattice_data_row.lemma)]
    tag_id = data_vocab['tag2id'][str(lattice_data_row.tag)]
    feat_ids = [data_vocab['feats2id'][f] for f in str(lattice_data_row.feats).split('|')]
    feat_ids += [data_vocab['feats2id']['_']] * (max_num_feats - len(feat_ids))
    values = [lattice_data_row.sent_id, lattice_data_row.token_id, lattice_data_row.analysis_id,
              lattice_data_row.morpheme_id]
    values += [lattice_data_row.is_gold and (infuse or not lattice_data_row.is_inf)]
    values += [form_id, lemma_id, tag_id]
    values += feat_ids
    return values


def _to_row_values(data_row, data_vocab):
    form_id = data_vocab['form2id'][str(data_row.form)]
    lemma_id = data_vocab['lemma2id'][str(data_row.lemma)]
    tag_id = data_vocab['tag2id'][str(data_row.tag)]
    feats_id = data_vocab['feats_str2id'][str(data_row.feats)]
    # morpheme_id = ['pref', 'host', 'suff'].index(row.morpheme_type) if morpheme_type else row.morpheme_id
    values = [data_row.sent_id, data_row.token_id, data_row.analysis_id, data_row.morpheme_id]
    values += [form_id, lemma_id, tag_id, feats_id]
    return values


# ldf - lattice data frame
def _get_token_samples(lattices_df, data_vocab):
    token_char_ids = {}
    column_names = ['sent_idx', 'token_idx', 'char_idx', 'token_id', 'char_id']
    token_row_values = [_to_tokens_row_values(lattice_data_row, data_vocab, token_char_ids)
                       for lattice_data_row in lattices_df.itertuples()]
    tokens_samples_df = pd.DataFrame([token_row for sent_token_rows in token_row_values
                                      for token_row in sent_token_rows], columns=column_names)

    # Token samples
    num_samples = tokens_samples_df.sent_idx.max()
    max_len = tokens_samples_df.token_idx.max()
    max_chars = tokens_samples_df.char_idx.max()
    token_samples = np.zeros((num_samples, max_len, max_chars, 2), dtype=np.int)
    sent_indices = tokens_samples_df.sent_idx.values - 1
    token_indices = tokens_samples_df.token_idx.values - 1
    char_indices = tokens_samples_df.char_idx.values - 1
    values = tokens_samples_df[['token_id', 'char_id']]
    token_samples[sent_indices, token_indices, char_indices] = values
    # Token and char lengths
    token_length_samples = np.zeros((num_samples, max_len, 2), dtype=np.int)
    char_lengths = tokens_samples_df.groupby(['sent_idx', 'token_idx'])[['char_idx']].max().squeeze()
    sent_indices = [v[0] - 1 for v in char_lengths.index.values]
    token_indices = [v[1] - 1 for v in char_lengths.index.values]
    token_length_samples[sent_indices, token_indices, 1] = char_lengths.values
    token_lengths = tokens_samples_df.groupby(['sent_idx'])[['token_idx']].max().squeeze()
    sent_indices = [v - 1 for v in token_lengths.index.values]
    token_length_samples[sent_indices, 0, 0] = token_lengths.values

    # num_sample (sent_idx.max()) may be greater than the actual number of samples if there are gaps in sent indices.
    # So we need to only keep the entries in the array that correspond to actual sentence indices.
    return (token_samples[tokens_samples_df.sent_idx.unique() - 1],
            token_length_samples[tokens_samples_df.sent_idx.unique() - 1])


def _get_lattice_analysis_samples(lattice_df, data_vocab, max_morphemes, max_feats_len, infuse=True):
    indices_column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morpheme_column_names = ['is_gold', 'form_id', 'lemma_id', 'tag_id']
    feat_column_names = [f'feat{i+1}_id' for i in range(max_feats_len)]
    column_names = indices_column_names + morpheme_column_names + feat_column_names
    lattice_values = [_to_lattice_row_values(lattice_data_row, max_feats_len, data_vocab, infuse)
                      for lattice_data_row in lattice_df.itertuples()]
    lattice_samples_df = pd.DataFrame(lattice_values, columns=column_names)

    # Morpheme samples
    num_samples = lattice_samples_df.sent_idx.max()
    max_len = lattice_samples_df.token_idx.max()
    max_analyses = lattice_samples_df.analysis_idx.max() + 1
    morpheme_len = len(morpheme_column_names) + len(feat_column_names)
    lattice_analysis_samples = np.zeros((num_samples, max_len, max_analyses, max_morphemes, morpheme_len), dtype=np.int)
    sent_indices = lattice_samples_df['sent_idx'].values - 1
    token_indices = lattice_samples_df['token_idx'].values - 1
    analysis_indices = lattice_samples_df['analysis_idx'].values
    morpheme_indices = lattice_samples_df['morpheme_idx'].values
    values = lattice_samples_df[morpheme_column_names + feat_column_names]
    lattice_analysis_samples[sent_indices, token_indices, analysis_indices, morpheme_indices] = values
    # Morpheme analysis lengths
    lattice_analysis_length_samples = np.zeros((num_samples, max_len), dtype=np.int)
    analysis_lengths_df = lattice_samples_df.groupby(['sent_idx', 'token_idx'])[['analysis_idx']].max().squeeze()
    sent_indices = [v[0] - 1 for v in analysis_lengths_df.index.values]
    token_indices = [v[1] - 1 for v in analysis_lengths_df.index.values]
    lattice_analysis_length_samples[sent_indices, token_indices] = analysis_lengths_df.values + 1

    # num_sample (sent_idx.max()) may be greater than the actual number of samples if there are gaps in sent indices.
    # So we need to only keep the entries in the array that correspond to actual sentence indices.
    return (lattice_analysis_samples[lattice_samples_df.sent_idx.unique() - 1],
            lattice_analysis_length_samples[lattice_samples_df.sent_idx.unique() - 1])


def _get_fixed_analysis_samples(analyses_df, data_vocab, max_morphemes):
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id', 'feats_id']
    column_names += morph_column_names
    morpheme_values = [_to_row_values(data_row, data_vocab) for data_row in analyses_df.itertuples()]
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
    morpheme_values = [_to_row_values(data_row, data_vocab) for data_row in analyses_df.itertuples()]
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


def _load_data(root_path, partition, data_type=None):
    dataset = {}
    for partition_type in partition:
        if data_type is not None:
            file_path = root_path / f'{partition_type}-{data_type}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}.lattices.csv'
        print(f'loading {file_path.stem}')
        dataset[partition_type] = pd.read_csv(str(file_path), index_col=0, keep_default_na=False)
        print(f'{file_path.stem} data size: {len(dataset[partition_type])}')
    return dataset


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
            morpheme_feats_dict = {f[0]: f[1] for f in [f.split("=") for f in morpheme_feats[morpheme_feats != '_']]}
            s = [f'{feat_name}={morpheme_feats_dict[feat_name]}' for feat_name in morpheme_feats_dict]
            token_feat_str_rows.append('|'.join(s))
        feat_str_rows.append(np.array(token_feat_str_rows))
    return np.stack(feat_str_rows)


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


def load_gold_vocab(root_path, la_name, tb_name):
    vocab_dir_path = root_path / la_name / tb_name / 'vocab'
    return _load_vocab(vocab_dir_path)


def load_gold_multi_vocab(root_path, la_name, tb_name, multi_tag_level):
    vocab_dir_path = root_path / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag' / 'vocab'
    return _load_vocab(vocab_dir_path)


def load_lattices_vocab(root_path, la_name, tb_name, ma_name):
    vocab_dir_path = root_path / la_name / tb_name / 'lattice' / ma_name / 'vocab'
    return _load_vocab(vocab_dir_path)


def load_gold_ft_emb(root_path, ft_root_path, data_vocab, la_name, tb_name):
    vocab_dir_path = root_path / la_name / tb_name / 'vocab'
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    return _load_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def load_gold_multi_ft_emb(root_path, ft_root_path, data_vocab, la_name, tb_name, multi_tag_level):
    vocab_dir_path = root_path / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag' / 'vocab'
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    return _load_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def load_lattice_ft_emb(root_path, ft_root_path, data_vocab, la_name, tb_name, ma_name):
    vocab_dir_path = root_path / la_name / tb_name / 'lattice' / ma_name / 'vocab'
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    return _load_morpheme_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def load_lattices_data_samples(root_path, partition, la_name, tb_name, ma_name):
    data_dir = root_path / la_name / tb_name / 'lattice' / ma_name
    lattices_dataset = _load_data(data_dir, partition)
    data_vocab = load_lattices_vocab(root_path, la_name, tb_name, ma_name)
    token_samples = {t: _get_token_samples(lattices_dataset[t], data_vocab) for t in lattices_dataset}

    # All variable sized attributes such as the number of morpheme per analysis or number of features per morpheme
    # must be the same across all partitions (train, dev, test) so all partition arrays are the same fixed size.
    max_morphemes = {t: lattices_dataset[t].morpheme_id.max() + 1 for t in partition}
    max_feats_len = {t: max(_get_feats_len(lattices_dataset[t].feats.values)) for t in partition}

    morph_samples = {t: _get_lattice_analysis_samples(lattices_dataset[t], data_vocab, max_morphemes[partition[-1]],
                                                      max_feats_len[partition[-1]]) for t in lattices_dataset}
    return token_samples, morph_samples, data_vocab


def load_gold_data_samples(root_path, partition, la_name, tb_name):
    data_dir = root_path / la_name / tb_name
    gold_dataset = _load_data(data_dir, partition, 'gold')
    data_vocab = load_gold_vocab(root_path, la_name, tb_name)
    token_samples = {t: _get_token_samples(gold_dataset[t], data_vocab) for t in gold_dataset}
    max_morphemes = {t: gold_dataset[t].morpheme_id.max() + 1 for t in partition}
    morph_samples = {t: _get_var_morpheme_samples(gold_dataset[t], data_vocab, max_morphemes[partition[-1]])
                        for t in gold_dataset}
    return token_samples, morph_samples, data_vocab


def load_gold_multi_data_samples(root_path, partition, la_name, tb_name, multi_tag_level):
    data_dir = root_path / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag'
    gold_dataset = _load_data(data_dir, partition, f'gold-{multi_tag_level}')
    data_vocab = load_gold_multi_vocab(root_path, la_name, tb_name, multi_tag_level)
    token_samples = {t: _get_token_samples(gold_dataset[t], data_vocab) for t in gold_dataset}
    max_morphemes = {t: gold_dataset[t].morpheme_id.max() + 1 for t in partition}
    morph_samples = {t: _get_fixed_analysis_samples(gold_dataset[t], data_vocab, max_morphemes[partition[-1]])
                        for t in gold_dataset}
    return token_samples, morph_samples, data_vocab


def _save_lattices_vocab(root_path, partition, la_name, tb_name, ma_name):
    lattices_dataset, gold_dataset = tb_load_infused_lattices(root_path, partition, la_name, tb_name, ma_name)
    lattices_vocab = _get_vocab(lattices_dataset)
    gold_vocab = _get_vocab(gold_dataset)
    gold_lattices_vocab = _get_vocabs_union(lattices_vocab, gold_vocab)
    _save_vocab(root_path / root_path / la_name / tb_name / 'lattice' / ma_name / 'vocab', gold_lattices_vocab)


def _save_gold_vocab(root_path, partition, la_name, tb_name):
    gold_dataset = tb_load_gold(root_path, partition, la_name, tb_name)
    gold_vocab = _get_vocab(gold_dataset)
    _save_vocab(root_path / root_path / la_name / tb_name / 'vocab', gold_vocab)


def _save_gold_multi_vocab(root_path, partition, la_name, tb_name, multi_tag_level):
    gold_dataset = tb_load_gold_multi_tag(root_path, partition, la_name, tb_name, multi_tag_level)
    gold_vocab = _get_vocab(gold_dataset)
    _save_vocab(root_path / root_path / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag' / 'vocab', gold_vocab)


def _save_gold_ft_emb(root_path, ft_root_path, la_name, tb_name):
    vocab_dir_path = root_path / la_name / tb_name / 'vocab'
    data_vocab = _load_vocab(vocab_dir_path)
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    ft.ft_model = None
    _save_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def _save_gold_multi_ft_emb(root_path, ft_root_path, la_name, tb_name, multi_tag_level):
    vocab_dir_path = root_path / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag' / 'vocab'
    data_vocab = _load_vocab(vocab_dir_path)
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    ft.ft_model = None
    _save_token_ft_emb(vocab_dir_path, ft_model_path, data_vocab)


def _save_lattice_ft_emb(root_path, ft_root_path, la_name, tb_name, ma_name):
    vocab_dir_path = root_path / la_name / tb_name / 'lattice' / ma_name / 'vocab'
    data_vocab = _load_vocab(vocab_dir_path)
    ft_model_path = ft_root_path / f'models/cc.{la_name}.300.bin'
    _save_morpheme_ft_emb(root_path, ft_model_path, data_vocab)


def main():
    scheme = 'UD'
    # scheme = 'SPMRL'
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / f'dev/aseker00/modi/tb/{scheme}'
    ft_path = Path.home() / 'dev/aseker00/fastText'
    if scheme == 'UD':
        tb_names = {'he': 'HTB', 'tr': 'IMST'}
    else:
        tb_names = {'he': 'HEBTB'}
    ma_names = {'he': 'heblex', 'tr': 'trmorph2'}
    for la_name in ['he', 'tr']:
        tb_name = tb_names[la_name]
        ma_name = ma_names[la_name]
        _save_gold_vocab(root_path, partition, la_name, tb_name)
        _save_gold_ft_emb(root_path, ft_path, la_name, tb_name)
        _save_gold_multi_vocab(root_path, partition, la_name, tb_name, 'token')
        _save_gold_multi_ft_emb(root_path, ft_path, la_name, tb_name, 'token')
        if scheme == 'SPMRL':
            _save_gold_multi_vocab(root_path, partition, la_name, tb_name, 'morpheme-type')
            _save_gold_multi_ft_emb(root_path, ft_path, la_name, tb_name, 'morpheme-type')
        _save_lattices_vocab(root_path, partition, la_name, tb_name, ma_name)
        _save_lattice_ft_emb(root_path, ft_path, la_name, tb_name, ma_name)

        token_samples, morph_samples, data_vocab = load_lattices_data_samples(root_path, partition, la_name, tb_name, ma_name)
        for partition_type in partition:
            print(f'{len(token_samples[partition_type])} {partition_type} token samples, '
                  f'{len(morph_samples[partition_type])} {partition_type} morpheme samples')
        token_samples, morph_samples, data_vocab = load_gold_data_samples(root_path, partition, la_name, tb_name)
        for partition_type in partition:
            print(f'{len(token_samples[partition_type])} {partition_type} token samples, '
                  f'{len(morph_samples[partition_type])} {partition_type} morpheme samples')
        token_samples, morph_samples, data_vocab = load_gold_multi_data_samples(root_path, partition, la_name, tb_name, 'token')
        for partition_type in partition:
            print(f'{len(token_samples[partition_type])} {partition_type} token samples, '
                  f'{len(morph_samples[partition_type])} {partition_type} morpheme samples')
        if scheme == 'SPMRL':
            token_samples, morph_samples, data_vocab = load_gold_multi_data_samples(root_path, partition, la_name, tb_name, 'morpheme-type')
            for partition_type in partition:
                print(f'{len(token_samples[partition_type])} {partition_type} token samples, '
                      f'{len(morph_samples[partition_type])} {partition_type} morpheme samples')


if __name__ == '__main__':
    main()
