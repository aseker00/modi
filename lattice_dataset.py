from dataset import *
import spmrl_treebank as tb
from pathlib import Path


def get_lattices_vocab(lattices, gold_lattices):
    tokens, forms, lemmas, tags, feats = set(), set(), set(), set(), set()
    for dataset in [lattices, gold_lattices]:
        for partition_type in dataset:
            for df in dataset[partition_type]:
                tokens.update(set(df.token))
                forms.update(set(df.form))
                lemmas.update(set(df.lemma))
                tags.update(set(df.tag))
                feats.update(set(df.feats))
    chars = set([c for t in tokens for c in list(t)])
    return to_vocab(tokens, chars, forms, lemmas, tags, feats)


def save_lattices_vocab(root_path, partition):
    dataset = tb.load_dataset(root_path, partition, 'lattices')
    gold_dataset = tb.load_dataset(root_path, partition, 'gold-lattices')
    vocab = get_lattices_vocab(dataset, gold_dataset)
    save_vocab(root_path / 'vocab', vocab)


def to_morpheme_row(row, max_num_feats, vocab, infuse):
    form_id = vocab['form2id'][row.form]
    lemma_id = vocab['lemma2id'][row.lemma]
    tag_id = vocab['tag2id'][row.tag]
    feat_ids = [vocab['feats2id'][f] for f in row.feats.split('|')]
    feat_ids += [vocab['feats2id']['_']] * (max_num_feats - len(feat_ids))
    values = [row.sent_id, row.token_id, row.analysis_id, row.morpheme_id]
    values += [row.is_gold and not row.is_dup and (infuse or not row.is_inf)]
    values += [form_id, lemma_id, tag_id]
    values += feat_ids
    return values


def get_uninf_lattices(df, vocab, max_morphemes):
    return get_lattices(df, vocab, max_morphemes, False)


def get_inf_lattices(df, vocab, max_morphemes):
    return get_lattices(df, vocab, max_morphemes, True)


def get_lattices(df, vocab, max_morphemes, infuse):
    max_feats_len = max(get_feats_len(df.feats.values))
    column_names = ['sent_idx', 'token_idx', 'analysis_idx', 'morpheme_idx']
    gold_column_names = ['is_gold']
    morph_column_names = ['form_id', 'lemma_id', 'tag_id']
    feat_column_names = [f'feat{i+1}_id' for i in range(max_feats_len)]
    column_names += gold_column_names
    column_names += morph_column_names
    column_names += feat_column_names
    sample_rows = [to_morpheme_row(row, max_feats_len, vocab, infuse) for row in df.itertuples() ]
    samples_df = pd.DataFrame(sample_rows, columns=column_names)
    num_samples = samples_df.sent_idx.max()
    max_len = samples_df.token_idx.max()
    max_analyses = samples_df.analysis_idx.max() + 1
    morpheme_len = len(morph_column_names) + len(gold_column_names) + len(feat_column_names)
    # Samples
    samples_arr = np.zeros((num_samples, max_len, max_analyses, max_morphemes, morpheme_len), dtype=np.int)
    sent_indices = samples_df['sent_idx'].values - 1
    token_indices = samples_df['token_idx'].values - 1
    analysis_indices = samples_df['analysis_idx'].values
    morpheme_indices = samples_df['morpheme_idx'].values
    values = samples_df[gold_column_names + morph_column_names + feat_column_names]
    samples_arr[sent_indices, token_indices, analysis_indices, morpheme_indices] = values
    # Analysis lengths
    analysis_length_arr = np.zeros((num_samples, max_len), dtype=np.int)
    analysis_lengths = samples_df.groupby(['sent_idx', 'token_idx'])[['analysis_idx']].max().squeeze()
    sent_indices = [v[0] - 1 for v in analysis_lengths.index.values]
    token_indices = [v[1] - 1 for v in analysis_lengths.index.values]
    analysis_length_arr[sent_indices, token_indices] = analysis_lengths.values + 1
    return samples_arr[samples_df.sent_idx.unique() - 1], analysis_length_arr[samples_df.sent_idx.unique() - 1]


def load_inf_lattices(root_path, partition):
    return load_data(root_path, partition, 'lattices-inf', get_inf_lattices)


def to_token_lattice(token_lattice_ids, vocab):
    token_form_ids = token_lattice_ids[:, :, 0]
    token_lemma_ids = token_lattice_ids[:, :, 1]
    token_tag_ids = token_lattice_ids[:, :, 2]
    token_feat_ids = token_lattice_ids[:, :, 3:]
    token_forms = to_form_vec(token_form_ids, vocab)
    token_lemmas = to_lemma_vec(token_lemma_ids, vocab)
    token_tags = to_tag_vec(token_tag_ids, vocab)
    token_feats_str = feats_to_str(to_feat_vec(token_feat_ids, vocab))
    return np.stack([token_forms, token_lemmas, token_tags, token_feats_str], axis=1)


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb'
    ft_root_path = Path.home() / 'dev/aseker00/fasttext'
    save_lattices_vocab(root_path / 'lattice', partition)
    save_ft_vec(root_path / 'lattice/vocab', ft_root_path)
    token_samples, lattice_samples, vocab = load_inf_lattices(root_path / 'lattice', partition)


if __name__ == '__main__':
    main()
