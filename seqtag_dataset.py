from pathlib import Path
from dataset import *
import treebank as tb


def save_gold_vocab(root_path, partition, tag_type):
    gold_lattices = tb.load_lattices_dataset(root_path, partition, tag_type)
    vocab = get_vocab(gold_lattices)
    save_vocab(root_path / 'vocab', vocab)


def to_morpheme_row(row, vocab, morpheme_type):
    form_id = vocab['form2id'][row.form]
    lemma_id = vocab['lemma2id'][row.lemma]
    tag_id = vocab['tag2id'][row.tag]
    feats_id = vocab['feats_str2id'][row.feats]
    morpheme_id = ['pref', 'host', 'suff'].index(row.morpheme_type) if morpheme_type else row.morpheme_id
    values = [row.sent_id, row.token_id, row.analysis_id, morpheme_id]
    values += [form_id, lemma_id, tag_id, feats_id]
    return values


def get_fixed_arr(df, vocab, max_morphemes):
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id', 'feats_id']
    column_names += morph_column_names
    sample_rows = [to_morpheme_row(row, vocab, True) for row in df.itertuples()]
    samples_df = pd.DataFrame(sample_rows, columns=column_names)
    num_samples = samples_df.sent_idx.max()
    max_len = samples_df.token_idx.max()
    default_morph_values = [vocab['form2id']['_'], vocab['lemma2id']['_'], vocab['tag2id']['_'], vocab['feats2id']['_']]
    samples_arr = np.array(default_morph_values, dtype=np.int)
    samples_arr = np.tile(samples_arr, (num_samples, max_len, max_morphemes, 1))
    sent_indices = samples_df['sent_idx'].values - 1
    token_indices = samples_df['token_idx'].values - 1
    morpheme_indices = samples_df['morpheme_idx'].values
    # values = samples_df[['form_id', 'lemma_id', 'tag_id', 'feats_id']].values
    values = samples_df[morph_column_names].values
    samples_arr[sent_indices, token_indices, morpheme_indices] = values
    # Set <PAD>
    # Find sentence boundary indices - this is used to get the number of tokens in each sentence
    token_mask = [bool(sent_indices[i] != sent_indices[i + 1]) for i in range(len(sent_indices) - 1)] + [True]
    # Use sentence boundary indices as start position for filling token indices
    fill_token_indices = [ii for i in token_indices[token_mask] for ii in range(i + 1, max_len)]
    # Now construct the sentence indices corresponding to the token indices
    fill_sent_indices = samples_df.sent_idx.unique() - 1
    fill_sent_indices = [fill_sent_indices[j].item() for j, i in enumerate(token_indices[token_mask])
                         for ii in range(i + 1, max_len)]
    samples_arr[fill_sent_indices, fill_token_indices] = 0
    return samples_arr[samples_df.sent_idx.unique() - 1]


def get_var_arr(df, vocab, max_morphemes):
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id', 'feats_id']
    column_names += morph_column_names
    sample_rows = [to_morpheme_row(row, vocab, False) for row in df.itertuples()]
    samples_df = pd.DataFrame(sample_rows, columns=column_names)
    max_sample = samples_df.sent_idx.max()
    max_len = samples_df.token_idx.max()
    samples_arr = np.zeros((max_sample, max_len, max_morphemes + 1, len(morph_column_names)), dtype=np.int)
    sent_indices = samples_df['sent_idx'].values - 1
    token_indices = samples_df['token_idx'].values - 1
    morpheme_indices = samples_df['morpheme_idx'].values
    # values = samples_df[['form_id', 'lemma_id', 'tag_id', 'feats_id']].values
    values = samples_df[morph_column_names].values
    samples_arr[sent_indices, token_indices, morpheme_indices] = values

    # Set <EOT>
    # Find sentence boundary indices - this is used to get the number of tokens in each sentence
    token_mask = [bool(sent_indices[i] != sent_indices[i + 1]) for i in range(len(sent_indices) - 1)] + [True]
    # Use sentence boundary indices as start position for filling token indices
    fill_token_indices = [ii for i in token_indices[token_mask] for ii in range(i + 1, max_len)]
    # (max_sample > # of samples) since some ZVL samples were filtered, so you have to map to the correct sentence id
    fill_sent_indices = samples_df.sent_idx.unique() - 1
    # Now construct the sentence indices corresponding to the token indices
    fill_sent_indices = [fill_sent_indices[j].item() for j, i in enumerate(token_indices[token_mask]) for ii in
                         range(i + 1, max_len)]
    # Fill Values
    samples_arr[fill_sent_indices, fill_token_indices] = 0

    # Find token boundary indices - this is used to get number of morphemes in each token analysis
    token_mask = [bool(token_indices[i] != token_indices[i + 1]) for i in range(len(token_indices) - 1)] + [True]
    # Use token boundary indices to get the <EOT> morpheme indices (which are zero based), token indices and sentence
    # indices (which are 1 based)
    fill_morpheme_indices = morpheme_indices[token_mask] + 1
    fill_token_indices = token_indices[token_mask]
    fill_sent_indices = sent_indices[token_mask]
    # Fill values
    eot_values = [vocab['form2id']['<EOT>'], vocab['lemma2id']['<EOT>'], vocab['tag2id']['<EOT>'],
                  vocab['feats2id']['<EOT>']]
    samples_arr[fill_sent_indices, fill_token_indices, fill_morpheme_indices] = eot_values
    return samples_arr[samples_df.sent_idx.unique() - 1]


def load_samples(root_path, partition, morph_level, seq_type):
    if morph_level == 'morpheme':
        if seq_type == 'fixed':
            return load_data_samples(root_path / morph_level, partition, 'gold-lattices', get_fixed_arr)
        elif seq_type == 'var':
            return load_data_samples(root_path / morph_level, partition, 'gold-lattices', get_var_arr)
    elif morph_level == 'morpheme-type':
        if seq_type == 'fixed':
            return load_data_samples(root_path / morph_level, partition, 'gold-lattices-multi', get_fixed_arr)
        elif seq_type == 'var':
            return load_data_samples(root_path / morph_level, partition, 'gold-lattices-multi', get_var_arr)


def token_tags_to_lattice_data(tokens, token_tags):
    column_names = ['from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'analysis_id',
                    'morpheme_id']
    # if (token_tags[:, 0] == '<EOT>').any():
    #     token_tags[:, 0][token_tags[:, 0] == '<EOT>'] = '_'
    token_tags[token_tags == '<EOT>'] = '<PAD>'
    token_tag_mask = token_tags != '<PAD>'
    token_tag_mask_indices = token_tag_mask.nonzero()
    token_tag_indices = token_tag_mask_indices[0]
    morpheme_indices = token_tag_mask_indices[1]
    tags = token_tags[token_tag_indices, morpheme_indices]
    forms = np.full_like(tags, fill_value='_')
    lemmas = np.full_like(tags, fill_value='_')
    feats = np.full_like(tags, fill_value='_')
    rows = []
    for i, token_tag_idx in enumerate(zip(token_tag_mask_indices[0], token_tag_mask_indices[1])):
        from_node_id = i
        to_node_id = i + 1
        token_idx = token_tag_idx[0]
        morpheme_idx = token_tag_idx[1]
        token = tokens[token_idx]
        form = forms[i]
        lemma = lemmas[i]
        tag = tags[i]
        feat = feats[i]
        row = [from_node_id, to_node_id, form, lemma, tag, feat, token_idx + 1, token, 0, morpheme_idx]
        rows.append(row)
    return pd.DataFrame(rows, columns=column_names)


def lattice_to_tags_arr(df):
    values = [x[1].tag.values for x in df.groupby('token_id')]
    max_len = max([len(a) for a in values])
    tags_arr = np.full_like(df.tag.values, shape=(len(values), max_len), fill_value='<PAD>')
    for i, a in enumerate(values):
        tags_arr[i, :len(a)] = a
    return tags_arr


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
    ft_root_path = Path.home() / 'dev/aseker00/fasttext'
    # save_gold_vocab(root_path / 'morpheme', partition, 'gold-lattices')
    save_ft_vec(root_path / 'morpheme-type/vocab', ft_root_path)
    # save_gold_vocab(root_path / 'morpheme-type', partition, 'gold-lattices-multi')
    # save_gold_vocab(root_path / 'analysis', partition, 'gold-lattices-multi')
    # token_samples, morph_samples, vocab = load_data_samples(root_path / 'morpheme-type', partition,
    #                                                         'gold-lattices-multi', get_fixed_morph_seq_samples)
    # token_samples, morph_samples, vocab = load_data_samples(root_path / 'morpheme', partition,
    #                                                         'gold-lattices', get_var_morph_seq_samples)
    # token_samples, morph_samples, vocab = load_samples(root_path, partition, 'morpheme', 'fixed')
    # token_samples, morph_samples, vocab = load_samples(root_path, partition, 'morpheme-type', 'var')


if __name__ == '__main__':
    main()
