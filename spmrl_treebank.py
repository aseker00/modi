from treebank_utils import *
from seqtag_treebank import *
from lattice_treebank import *
from pathlib import Path


def build_spmrl_sample(sent_id, lattice, tokens, column_names, is_gold):
    for morpheme in lattice:
        morpheme[0] = int(morpheme[0])
        morpheme[1] = int(morpheme[1])
        del morpheme[5]
        morpheme[-1] = int(morpheme[-1])
        morpheme.append(tokens[morpheme[-1]])
        morpheme.append(is_gold)
        morpheme.insert(0, sent_id)
    return pd.DataFrame(lattice, columns=column_names)


def load_spmrl_partition(lattices_file_path, tokens_file_path, column_names, is_gold):
    partition = []
    lattice_sentences = split_sentences(lattices_file_path)
    token_sentences = split_sentences(tokens_file_path)
    for i, (lattice, tokens) in enumerate(zip(lattice_sentences, token_sentences)):
        sent_id = i + 1
        tokens = {j + 1: t for j, t in enumerate(tokens)}
        lattice = [line.split() for line in lattice]
        df = build_spmrl_sample(sent_id, lattice, tokens, column_names, is_gold)
        partition.append(df)
    return partition


def load_spmrl_treebank_lattices(tb_path, partition, column_names, lang, tb_name, spmrl_type):
    treebank = {}
    for partition_type in partition:
        file_name = f'{partition_type}_{tb_name}'.lower()
        print(f'loading {file_name} dataset')
        if spmrl_type == 'lattices':
            is_gold = False
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.lattices'
        else:
            is_gold = True
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}-gold.lattices'
        tokens_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.tokens'
        lattices = load_spmrl_partition(lattices_path, tokens_path, column_names, is_gold)
        print(f'{partition_type} lattices: {len(lattices)}')
        treebank[partition_type] = lattices
    return treebank


def get_sent_indices_to_remove(dataset, tags):
    return [i + 1 for i, df in enumerate(dataset) if set(df.tag).intersection(tags)]


def remove_from_partition(partition, tags):
    indices = get_sent_indices_to_remove(partition[1], tags)
    lattices = [df for df in partition[0] if df.sent_id.unique() not in indices]
    gold_lattices = [df for df in partition[1] if df.sent_id.unique() not in indices]
    print(f'{len(indices)} removed samples')
    return lattices, gold_lattices


def save_gold(tb_path, root_path, partition, lang, la_name, tb_name, remove_zvl=True):
    gold_lattices = load_spmrl_treebank_lattices(tb_path, partition, lattice_fields, lang, tb_name, 'gold-lattices')
    if remove_zvl:
        indices = get_sent_indices_to_remove(gold_lattices['train'], ['ZVL'])
        gold_lattices['train'] = [df for df in gold_lattices['train'] if df.sent_id.unique() not in indices]
    gold_dataset = get_dataset(gold_lattices)
    save_dataset(root_path / la_name, gold_dataset, 'gold-lattices')


def save_gold_morpheme_types(root_path, partition, la_name):
    gold_dataset = load_dataset(root_path / la_name, partition, 'gold-lattices')
    add_morpheme_type(gold_dataset)
    save_dataset(root_path / la_name / 'seq', gold_dataset, 'gold-lattices-type')


def save_gold_morpheme_multi_tag(root_path, partition, la_name):
    gold_dataset = load_dataset(root_path / la_name / 'seq', partition, 'gold-lattices-type')
    grouped_gold_dataset = get_grouped_morpheme_type_dataset(gold_dataset, lattice_fields)
    save_dataset(root_path / la_name / 'seq' / 'morpheme-multi-tag', grouped_gold_dataset, 'gold-lattices-multi')


def save_gold_token_super_tag(root_path, partition, la_name):
    gold_dataset = load_dataset(root_path / la_name, partition, 'gold-lattices')
    grouped_gold_dataset = get_grouped_analysis_dataset(gold_dataset, lattice_fields)
    save_dataset(root_path / la_name / 'seq' / 'token-super-tag', grouped_gold_dataset, 'gold-lattices-super')


def save_lattices(tb_path, root_path, partition, lang, la_name, tb_name, remove_zvl=True):
    lattices = load_spmrl_treebank_lattices(tb_path, partition, lattice_fields, lang, tb_name, 'lattices')
    gold_lattices = load_spmrl_treebank_lattices(tb_path, partition, lattice_fields, lang, tb_name, 'gold-lattices')
    if remove_zvl:
        indices = get_sent_indices_to_remove(gold_lattices['train'], ['ZVL'])
        lattices['train'] = [df for df in lattices['train'] if df.sent_id.unique() not in indices]
    dataset = get_dataset(lattices)
    save_dataset(root_path / la_name / 'lattice', dataset, 'lattices')


def save_infused(root_path, partition, la_name):
    gold_dataset = load_dataset(root_path / la_name, partition, 'gold-lattices')
    dataset = load_dataset(root_path / la_name / 'lattice', partition, 'lattices')
    infused_dataset = infuse(dataset, gold_dataset)
    save_dataset(root_path / la_name / 'lattice', infused_dataset, 'lattices-inf')


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl'
    tb_path = Path.home() / 'dev/onlplab/HebrewResources'
    langs = {'heb': 'Hebrew'}
    tb_names = {'heb': 'hebtb'}
    for la_name in ['heb']:
        lang = langs[la_name]
        tb_name = tb_names[la_name]
        save_gold(tb_path, root_path, partition, lang, la_name, tb_name)
        save_gold_morpheme_types(root_path, partition, la_name)
        save_gold_morpheme_multi_tag(root_path, partition, la_name)
        save_gold_token_super_tag(root_path, partition, la_name)
        save_lattices(tb_path, root_path, partition, lang, la_name, tb_name)
        save_infused(root_path, partition, la_name)


if __name__ == '__main__':
    main()
