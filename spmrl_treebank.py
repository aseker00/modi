from treebank_utils import _lattice_fields, _to_data_lattices, _save_data_lattices, _load_data_lattices,\
    _split_sentences
from seqtag_treebank import *
from lattice_treebank import *
from pathlib import Path


def _build_spmrl_sample(sent_id, lattice, tokens, column_names, is_gold):
    for morpheme in lattice:
        morpheme[0] = int(morpheme[0])
        morpheme[1] = int(morpheme[1])
        del morpheme[5]
        morpheme[-1] = int(morpheme[-1])
        morpheme.append(tokens[morpheme[-1]])
        morpheme.append(is_gold)
        morpheme.insert(0, sent_id)
    return pd.DataFrame(lattice, columns=column_names)


def _load_spmrl_conll_partition(lattices_file_path, tokens_file_path, column_names, is_gold):
    partition = []
    lattice_sentences = _split_sentences(lattices_file_path)
    token_sentences = _split_sentences(tokens_file_path)
    for i, (lattice, tokens) in enumerate(zip(lattice_sentences, token_sentences)):
        sent_id = i + 1
        tokens = {j + 1: t for j, t in enumerate(tokens)}
        lattice = [line.split() for line in lattice]
        df = _build_spmrl_sample(sent_id, lattice, tokens, column_names, is_gold)
        partition.append(df)
    return partition


def _load_spmrl_conll(tb_path, partition, column_names, lang, tb_name, is_gold=False):
    treebank = {}
    for partition_type in partition:
        file_name = f'{partition_type}_{tb_name}'.lower()
        print(f'loading {file_name} dataset')
        if is_gold:
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}-gold.lattices'
        else:
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.lattices'
        tokens_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.tokens'
        lattices = _load_spmrl_conll_partition(lattices_path, tokens_path, column_names, is_gold)
        print(f'{partition_type} lattices: {len(lattices)}')
        treebank[partition_type] = lattices
    return treebank


def _get_sent_indices_to_remove(dataset, tags):
    return [i + 1 for i, df in enumerate(dataset) if set(df.tag).intersection(tags)]


def _remove_from_partition(partition, tags):
    indices = _get_sent_indices_to_remove(partition[1], tags)
    lattices = [df for df in partition[0] if df.sent_id.unique() not in indices]
    gold_lattices = [df for df in partition[1] if df.sent_id.unique() not in indices]
    print(f'{len(indices)} removed samples')
    return lattices, gold_lattices


def _save_gold(tb_path, root_path, partition, lang, la_name, tb_name, remove_zvl=True):
    gold_lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name, True)
    if remove_zvl:
        indices = _get_sent_indices_to_remove(gold_lattices['train'], ['ZVL'])
        gold_lattices['train'] = [df for df in gold_lattices['train'] if df.sent_id.unique() not in indices]
    gold_dataset = _to_data_lattices(gold_lattices)
    _save_data_lattices(root_path / la_name, gold_dataset, 'gold')


def _save_gold_morpheme_types(root_path, partition, la_name):
    gold_dataset = _load_data_lattices(root_path / la_name, partition, 'gold')
    add_morpheme_type(gold_dataset)
    _save_data_lattices(root_path / la_name / 'seq', gold_dataset, 'gold-type')


def _save_gold_morpheme_multi_tag(root_path, partition, la_name):
    gold_dataset = _load_data_lattices(root_path / la_name / 'seq', partition, 'gold-type')
    grouped_gold_dataset = get_grouped_morpheme_type_dataset(gold_dataset, _lattice_fields)
    _save_data_lattices(root_path / la_name / 'seq' / 'morpheme-multi-tag', grouped_gold_dataset, 'gold-multi')


def _save_gold_token_super_tag(root_path, partition, la_name):
    gold_dataset = _load_data_lattices(root_path / la_name, partition, 'gold')
    grouped_gold_dataset = get_grouped_analysis_dataset(gold_dataset, _lattice_fields)
    _save_data_lattices(root_path / la_name / 'seq' / 'token-super-tag', grouped_gold_dataset, 'gold-super')


def _save_uninfused_lattices(tb_path, root_path, partition, lang, la_name, tb_name, remove_zvl=True):
    lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name)
    gold_lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name, True)
    if remove_zvl:
        indices = _get_sent_indices_to_remove(gold_lattices['train'], ['ZVL'])
        lattices['train'] = [df for df in lattices['train'] if df.sent_id.unique() not in indices]
    dataset = _to_data_lattices(lattices)
    _save_data_lattices(root_path / la_name / 'lattice', dataset, 'uninf')


def _save_lattices(root_path, partition, la_name):
    dataset, gold_dataset = load_uninfused_lattices(root_path, partition, la_name)
    infused_dataset = infuse_tb_lattices(dataset, gold_dataset)
    _save_data_lattices(root_path / la_name / 'lattice', infused_dataset)


def load_lattices(root_path, partition, la_name):
    gold_dataset = _load_data_lattices(root_path / la_name, partition, 'gold')
    infused_dataset = _load_data_lattices(root_path / la_name / 'lattice', partition)
    return infused_dataset, gold_dataset


def load_uninfused_lattices(root_path, partition, la_name):
    gold_dataset = _load_data_lattices(root_path / la_name, partition, 'gold')
    dataset = _load_data_lattices(root_path / la_name / 'lattice', partition, 'uninf')
    return dataset, gold_dataset


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl'
    tb_path = Path.home() / 'dev/onlplab/HebrewResources'
    langs = {'heb': 'Hebrew'}
    tb_names = {'heb': 'hebtb'}
    for la_name in ['heb']:
        lang = langs[la_name]
        tb_name = tb_names[la_name]
        _save_gold(tb_path, root_path, partition, lang, la_name, tb_name)
        _save_gold_morpheme_types(root_path, partition, la_name)
        _save_gold_morpheme_multi_tag(root_path, partition, la_name)
        _save_gold_token_super_tag(root_path, partition, la_name)
        _save_uninfused_lattices(tb_path, root_path, partition, lang, la_name, tb_name)
        _save_lattices(root_path, partition, la_name)


if __name__ == '__main__':
    main()
