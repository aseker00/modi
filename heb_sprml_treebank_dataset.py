from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
import heb_spmrl_treebank as tb
import fasttext_emb as ft


def get_analysis(morphemes):
    analysis = defaultdict(list)
    for morpheme in morphemes:
        # morpheme_id = int(morpheme[0])
        morpheme_type = morpheme[0]
        # morpheme_from_node_id = int(morpheme[1])
        # morpheme_to_node_id = int(morpheme[2])
        morpheme_form = morpheme[3]
        morpheme_lemma = morpheme[4]
        morpheme_tag = morpheme[5]
        morpheme_feats = morpheme[6]
        m = (morpheme_form, morpheme_lemma, morpheme_tag, morpheme_feats)
        analysis[morpheme_type].append(m)
    return analysis


def _get_token_samples(partition):
    lattice_samples = defaultdict(list)
    gold_samples = defaultdict(list)
    for partition_type in partition:
        lattices, gold_lattices = partition[partition_type]
        for sentences, samples in zip([lattices, gold_lattices], [lattice_samples, gold_samples]):
            for sent_id in sentences:
                raw_sent_sample = []
                for sent_token_id in lattices[sent_id]:
                    token_morphemes = next(iter(lattices[sent_id][sent_token_id].values()))
                    token = token_morphemes[0][-1]
                    token_chars = list(token)
                    token_analyses = {}
                    token_analysis = get_analysis(token_morphemes)
                    for morpheme_type in ['pref', 'host', 'suff']:
                        if morpheme_type not in token_analysis:
                            morphs = ('_', '_', '_', '_')
                            token_analysis[morpheme_type].append(morphs)
                    for morpheme_type in token_analysis:
                        morphemes = token_analysis[morpheme_type]
                        m_forms = [m[0] for m in morphemes]
                        m_lemmas = [m[1] for m in morphemes]
                        m_tags = [m[2] for m in morphemes]
                        m_feats = [m[3] for m in morphemes]
                        token_analyses[morpheme_type] = [(('-'.join(m_forms), '-'.join(m_lemmas), '-'.join(m_tags),
                                                           '-'.join(m_feats)))]
                    raw_sent_sample.append((token, token_chars, token_analyses))
                samples[partition_type].append(raw_sent_sample)
    return lattice_samples, gold_samples


def _get_morpheme_samples(partition):
    lattice_samples = defaultdict(list)
    gold_samples = defaultdict(list)
    for partition_type in partition:
        lattices, gold_lattices = partition[partition_type]
        for sentences, samples in zip([lattices, gold_lattices], [lattice_samples, gold_samples]):
            for sent_id in sentences:
                raw_sent_sample = []
                for sent_token_id in lattices[sent_id]:
                    token_morphemes = next(iter(lattices[sent_id][sent_token_id].values()))
                    token = token_morphemes[0][-1]
                    token_chars = list(token)
                    token_analyses = get_analysis(token_morphemes)
                    raw_sent_sample.append((token, token_chars, token_analyses))
                samples[partition_type].append(raw_sent_sample)
    return lattice_samples, gold_samples


def _extract_vocab(lattice_samples, gold_samples):
    tokens, chars, forms, lemmas, tags, feats = set(), set(), set(), set(), set(), set()
    for samples in [lattice_samples, gold_samples]:
        for name in samples:
            for sent_sample in samples[name]:
                for (token, token_chars, token_morphemes) in sent_sample:
                    tokens.add(token)
                    chars.update(token_chars)
                    for morpheme_type in token_morphemes:
                        morphemes = token_morphemes[morpheme_type]
                        forms.update([m[0] for m in morphemes])
                        lemmas.update([m[1] for m in morphemes])
                        tags.update([m[2] for m in morphemes])
                        feats.update([m[3] for m in morphemes])
    tokens = ['<PAD>'] + sorted(list(tokens))
    chars = ['<PAD>'] + sorted(list(chars))
    for m in [forms, lemmas, tags, feats]:
        if '_' in m:
            m.remove('_')
    forms = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(forms))
    lemmas = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(lemmas))
    tags = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(tags))
    feats = ['<PAD>', '<SOS>', '<EOT>', '_'] + sorted(list(feats))
    token2id = {v: i for i, v in enumerate(tokens)}
    char2id = {v: i for i, v in enumerate(chars)}
    form2id = {v: i for i, v in enumerate(forms)}
    lemma2id = {v: i for i, v in enumerate(lemmas)}
    tag2id = {v: i for i, v in enumerate(tags)}
    feats2id = {v: i for i, v in enumerate(feats)}
    return {'tokens': tokens, 'token2id': token2id, 'chars': chars, 'char2id': char2id,
            'forms': forms, 'form2id': form2id, 'lemmas': lemmas, 'lemma2id': lemma2id,
            'tags': tags, 'tag2id': tag2id, 'feats': feats, 'feats2id': feats2id}


def _save_vocab(root_dir_path, vocab, samples_type):
    vocab_dir_path = root_dir_path / 'treebank/spmrl/heb/vocab'
    with open(str(vocab_dir_path / 'chars.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['chars']])
    with open(str(vocab_dir_path / 'tokens.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['tokens']])
    with open(str(vocab_dir_path / f'{samples_type}_forms.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['forms']])
    with open(str(vocab_dir_path / f'{samples_type}_lemmas.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['lemmas']])
    with open(str(vocab_dir_path / f'{samples_type}_tags.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['tags']])
    with open(str(vocab_dir_path / f'{samples_type}_feats.txt'), 'w') as f:
        f.writelines([f'{v}\n' for v in vocab['feats']])


def _load_vocab(root_dir_path, samples_type):
    vocab_dir_path = root_dir_path / 'treebank/spmrl/heb/vocab'
    with open(str(vocab_dir_path / 'chars.txt')) as f:
        chars = [line.strip() for line in f.readlines()]
    with open(str(vocab_dir_path / 'tokens.txt')) as f:
        tokens = [line.strip() for line in f.readlines()]
    with open(str(vocab_dir_path / f'{samples_type}_forms.txt')) as f:
        forms = [line.strip() for line in f.readlines()]
    with open(str(vocab_dir_path / f'{samples_type}_lemmas.txt')) as f:
        lemmas = [line.strip() for line in f.readlines()]
    with open(str(vocab_dir_path / f'{samples_type}_tags.txt')) as f:
        tags = [line.strip() for line in f.readlines()]
    with open(str(vocab_dir_path / f'{samples_type}_feats.txt')) as f:
        feats = [line.strip() for line in f.readlines()]
    token2id = {v: i for i, v in enumerate(tokens)}
    char2id = {v: i for i, v in enumerate(chars)}
    form2id = {v: i for i, v in enumerate(forms)}
    lemma2id = {v: i for i, v in enumerate(lemmas)}
    tag2id = {v: i for i, v in enumerate(tags)}
    feats2id = {v: i for i, v in enumerate(feats)}
    return {'tokens': tokens, 'token2id': token2id, 'chars': chars, 'char2id': char2id,
            'forms': forms, 'form2id': form2id, 'lemmas': lemmas, 'lemma2id': lemma2id,
            'tags': tags, 'tag2id': tag2id, 'feats': feats, 'feats2id': feats2id}


def _save_ft_emb(root_dir_path, ft_root_dir_path, morpheme_vocab, token_vocab):
    vocab_dir_path = root_dir_path / 'treebank/spmrl/heb/vocab'
    ft_model_path = ft_root_dir_path / 'models/cc.he.300.bin'
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    morpheme_forms_vec_file_path = vocab_dir_path / 'morpheme_forms.vec'
    morpheme_lemmas_vec_file_path = vocab_dir_path / 'morpheme_lemmas.vec'
    token_forms_vec_file_path = vocab_dir_path / 'token_forms.vec'
    token_lemmas_vec_file_path = vocab_dir_path / 'token_lemmas.vec'
    if chars_vec_file_path.exists():
        chars_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, morpheme_vocab['chars'])
    if tokens_vec_file_path.exists():
        tokens_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, morpheme_vocab['tokens'])
    if morpheme_forms_vec_file_path.exists():
        morpheme_forms_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, morpheme_forms_vec_file_path, morpheme_vocab['forms'])
    if morpheme_lemmas_vec_file_path.exists():
        morpheme_lemmas_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, morpheme_lemmas_vec_file_path, morpheme_vocab['lemmas'])
    if token_forms_vec_file_path.exists():
        token_forms_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, token_forms_vec_file_path, token_vocab['forms'])
    if token_lemmas_vec_file_path.exists():
        token_lemmas_vec_file_path.unlink()
    ft.load_embedding_weight_matrix(ft_model_path, token_lemmas_vec_file_path, token_vocab['lemmas'])


def load_morpheme_ft_emb(root_dir_path, ft_root_dir_path, morpheme_vocab):
    vocab_dir_path = root_dir_path / 'treebank/spmrl/heb/vocab'
    ft_model_path = ft_root_dir_path / 'models/cc.he.300.bin'
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    morpheme_forms_vec_file_path = vocab_dir_path / 'morpheme_forms.vec'
    morpheme_lemmas_vec_file_path = vocab_dir_path / 'morpheme_lemmas.vec'
    return (ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, morpheme_vocab['chars']),
            ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, morpheme_vocab['tokens']),
            ft.load_embedding_weight_matrix(ft_model_path, morpheme_forms_vec_file_path, morpheme_vocab['forms']),
            ft.load_embedding_weight_matrix(ft_model_path, morpheme_lemmas_vec_file_path, morpheme_vocab['lemmas']))


def load_token_ft_emb(root_dir_path, ft_root_dir_path, token_vocab):
    vocab_dir_path = root_dir_path / 'treebank/spmrl/heb/vocab'
    ft_model_path = ft_root_dir_path / 'models/cc.he.300.bin'
    chars_vec_file_path = vocab_dir_path / 'chars.vec'
    tokens_vec_file_path = vocab_dir_path / 'tokens.vec'
    token_forms_vec_file_path = vocab_dir_path / 'token_forms.vec'
    token_lemmas_vec_file_path = vocab_dir_path / 'token_lemmas.vec'
    return (ft.load_embedding_weight_matrix(ft_model_path, chars_vec_file_path, token_vocab['chars']),
            ft.load_embedding_weight_matrix(ft_model_path, tokens_vec_file_path, token_vocab['tokens']),
            ft.load_embedding_weight_matrix(ft_model_path, token_forms_vec_file_path, token_vocab['forms']),
            ft.load_embedding_weight_matrix(ft_model_path, token_lemmas_vec_file_path, token_vocab['lemmas']))


def _load_data_arrays(vocab, samples):
    token2id = vocab['token2id']
    char2id = vocab['char2id']
    form2id = vocab['form2id']
    lemma2id = vocab['lemma2id']
    tag2id = vocab['tag2id']
    feats2id = vocab['feats2id']
    ds = defaultdict(list)
    for partition_type in samples:
        token_samples = []
        token_char_samples = []
        token_morph_samples = []
        for raw_sent_sample in samples[partition_type]:
            sample_token_ids = []
            sample_token_char_ids = []
            sample_token_morph_ids = []
            for (token, token_chars, token_morphemes) in raw_sent_sample:
                token_id = token2id[token]
                token_char_ids = [char2id[c] for c in token_chars]
                token_morph_ids = defaultdict(list)
                for morph_type in token_morphemes:
                    morphemes = token_morphemes[morph_type]
                    morpheme_type_id = ['pref', 'host', 'suff'].index(morph_type)
                    for m in morphemes:
                        morph_form_id = form2id[m[0]]
                        morph_lemma_id = lemma2id[m[1]]
                        morph_tag_id = tag2id[m[2]]
                        morph_feats_id = feats2id[m[3]]
                        m_ids = (morph_form_id, morph_lemma_id, morph_tag_id, morph_feats_id)
                        token_morph_ids[morpheme_type_id].append(m_ids)
                sample_token_ids.append(token_id)
                sample_token_char_ids.append(token_char_ids)
                token_form_ids = [m_ids[0] for morph_type in range(3) for m_ids in token_morph_ids[morph_type]]
                token_lemma_ids = [m_ids[1] for morph_type in range(3) for m_ids in token_morph_ids[morph_type]]
                token_tag_ids = [m_ids[2] for morph_type in range(3) for m_ids in token_morph_ids[morph_type]]
                token_feats_ids = [m_ids[3] for morph_type in range(3) for m_ids in token_morph_ids[morph_type]]
                sample_token_morph_ids.append(token_form_ids + token_lemma_ids + token_tag_ids + token_feats_ids)
            token_samples.append(sample_token_ids)
            token_char_samples.append(sample_token_char_ids)
            token_morph_samples.append(sample_token_morph_ids)
        token_arrs = [np.array(sample) for sample in token_samples]
        char_arrs = [[np.array(chars) for chars in sample] for sample in token_char_samples]
        max_morph_sample_len = max([len(m) for sample in token_morph_samples for m in sample])
        max_num_morphs = max_morph_sample_len // 4
        morph_arrs = [np.zeros((len(sample), max_morph_sample_len), dtype=np.int) for sample in token_morph_samples]
        for i, sample in enumerate(token_morph_samples):
            for j, morph_ids in enumerate(sample):
                num_morphs = len(morph_ids) // 4
                morph_arrs[i][j, 0 * max_num_morphs:0 * max_num_morphs + num_morphs] = morph_ids[0 * num_morphs:1 * num_morphs]
                morph_arrs[i][j, 1 * max_num_morphs:1 * max_num_morphs + num_morphs] = morph_ids[1 * num_morphs:2 * num_morphs]
                morph_arrs[i][j, 2 * max_num_morphs:2 * max_num_morphs + num_morphs] = morph_ids[2 * num_morphs:3 * num_morphs]
                morph_arrs[i][j, 3 * max_num_morphs:3 * max_num_morphs + num_morphs] = morph_ids[3 * num_morphs:4 * num_morphs]
        ds[partition_type] = (token_arrs, char_arrs, morph_arrs)
    return ds


def _load_tensor_dataset(data_arrays):
    max_tokens_num = max([token_arr.shape[0] for partition_type in data_arrays for token_arr in
                          data_arrays[partition_type][0]])
    max_chars_num = max([char_arr.shape[0] for partition_type in data_arrays for char_arrs in
                         data_arrays[partition_type][1] for char_arr in char_arrs])
    max_morph_num = max([morph_arrs.shape[1] for partition_type in data_arrays for morph_arrs in
                         data_arrays[partition_type][2]])
    morph_len = max_morph_num // 4
    ds = {}
    for partition_type in data_arrays:
        token_arrs = data_arrays[partition_type][0]
        token_lengths = []
        padded_tokens_arr = np.zeros((len(token_arrs), max_tokens_num), dtype=np.long)
        for i, j in enumerate(token_arrs):
            padded_tokens_arr[i][:len(j)] = j
            token_lengths.append(len(j))
        char_arrs = data_arrays[partition_type][1]
        char_lengths = np.zeros((len(char_arrs), max_tokens_num), dtype=np.long)
        padded_chars_arr = np.zeros((len(char_arrs), max_tokens_num, max_chars_num), dtype=np.long)
        for i, char_arr in enumerate(char_arrs):
            for j, k in enumerate(char_arr):
                padded_chars_arr[i][j][:len(k)] = k
                char_lengths[i][j] = len(k)
        morph_arrs = data_arrays[partition_type][2]
        padded_morphs_arr = np.zeros((len(morph_arrs), max_tokens_num, max_morph_num), dtype=np.long)
        for i, morph_arr in enumerate(morph_arrs):
            for j, k in enumerate(morph_arr):
                arr_len = len(k)
                arr_morph_len = arr_len // 4
                form_ids = k[0 * arr_morph_len:1 * arr_morph_len]
                lemma_ids = k[1 * arr_morph_len:2 * arr_morph_len]
                tag_ids = k[2 * arr_morph_len:3 * arr_morph_len]
                feat_ids = k[3 * arr_morph_len:4 * arr_morph_len]
                padded_morphs_arr[i][j][0 * morph_len:0 * morph_len + len(form_ids)] = form_ids
                padded_morphs_arr[i][j][1 * morph_len:1 * morph_len + len(lemma_ids)] = lemma_ids
                padded_morphs_arr[i][j][2 * morph_len:2 * morph_len + len(tag_ids)] = tag_ids
                padded_morphs_arr[i][j][3 * morph_len:3 * morph_len + len(feat_ids)] = feat_ids
        tokens_tensor = torch.from_numpy(padded_tokens_arr)
        token_lengths_tensor = torch.tensor(token_lengths)
        chars_tensor = torch.from_numpy(padded_chars_arr)
        char_lengths_tensor = torch.tensor(char_lengths)
        morphs_tensor = torch.from_numpy(padded_morphs_arr)
        ds[partition_type] = TensorDataset(tokens_tensor, token_lengths_tensor, chars_tensor, char_lengths_tensor,
                                           morphs_tensor)
    return ds


def _save_token_vocab(root_dir_path, partition):
    lattice_samples, gold_samples = _get_token_samples(partition)
    token_vocab = _extract_vocab(lattice_samples, gold_samples)
    _save_vocab(root_dir_path, token_vocab, 'token')


def load_token_dataset(root_dir_path, partition):
    lattice_samples, gold_samples = _get_token_samples(partition)
    token_vocab = _load_vocab(root_dir_path, 'token')
    lattice_arrays = _load_data_arrays(token_vocab, lattice_samples)
    gold_lattice_arrays = _load_data_arrays(token_vocab, gold_samples)
    lattice_dataset = _load_tensor_dataset(lattice_arrays)
    gold_lattice_dataset = _load_tensor_dataset(gold_lattice_arrays)
    return token_vocab, lattice_dataset, gold_lattice_dataset


def _save_morpheme_dataset(root_dir_path, partition):
    lattice_samples, gold_samples = _get_morpheme_samples(partition)
    morpheme_vocab = _extract_vocab(lattice_samples, gold_samples)
    _save_vocab(root_dir_path, morpheme_vocab, 'morpheme')


def load_morpheme_dataset(root_dir_path, partition):
    lattice_samples, gold_samples = _get_morpheme_samples(partition)
    morpheme_vocab = _load_vocab(root_dir_path, 'morpheme')
    lattice_arrays = _load_data_arrays(morpheme_vocab, lattice_samples)
    gold_lattice_arrays = _load_data_arrays(morpheme_vocab, gold_samples)
    lattice_dataset = _load_tensor_dataset(lattice_arrays)
    gold_lattice_dataset = _load_tensor_dataset(gold_lattice_arrays)
    return morpheme_vocab, lattice_dataset, gold_lattice_dataset


def save_data(root_dir_path):
    partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
    _save_token_vocab(root_dir_path, partition)
    _save_morpheme_dataset(root_dir_path, partition)


def save_ft_vec(root_dir_path, ft_root_dir_path):
    morpheme_vocab = _load_vocab(root_dir_path, 'morpheme')
    token_vocab = _load_vocab(root_dir_path, 'token')
    _save_ft_emb(root_dir_path, ft_root_dir_path, morpheme_vocab, token_vocab)


def main():
    root_dir_path = Path.home() / 'dev/aseker00/modi'
    ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
    save_data(root_dir_path)
    save_ft_vec(root_dir_path, ft_root_dir_path)
    partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
    token_vocab, lattice_dataset, gold_lattice_dataset = load_token_dataset(root_dir_path, partition)
    morpheme_vocab, lattice_dataset, gold_lattice_dataset = load_morpheme_dataset(root_dir_path, partition)
    char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = load_token_ft_emb(root_dir_path, ft_root_dir_path,
                                                                             token_vocab)
    char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = load_morpheme_ft_emb(root_dir_path, ft_root_dir_path,
                                                                                morpheme_vocab)


if __name__ == '__main__':
    main()
