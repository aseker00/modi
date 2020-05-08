from treebank_utils import _split_sentences, _lattice_fields, _to_data_lattices, _save_data_lattices,\
    _load_data_lattices, _validate_data_lattices, _normalize_lattice
from seqtag_treebank import *
from lattice_treebank import *
from pathlib import Path


def _build_ud_sample(sent_id, ud_lattice, column_names):
    tokens = {}
    lattice = []
    for morpheme in ud_lattice:
        if len(morpheme[0].split('-')) == 2:
            from_node_id, to_node_id = (int(v) for v in morpheme[0].split('-'))
            token = morpheme[1]
            tokens[to_node_id] = token
        else:
            is_gold = False
            if len(morpheme) == 10:
                morpheme[0] = int(morpheme[0])
                morpheme.insert(0, morpheme[0] - 1)
                del morpheme[7]
                del morpheme[7]
                is_gold = True
            elif len(morpheme) == 9:
                morpheme[0] = int(morpheme[0])
                morpheme[1] = int(morpheme[1])
                # if 'goldId' in morpheme[7]:
                #     is_gold = True
            else:
                raise Exception(f'sent {sent_id} invalid morpheme: {morpheme}')
            morpheme_token_node_id = morpheme[1]
            token = morpheme[2]
            token_node_id = 0
            token_id = 0
            for i, node_id in enumerate(tokens):
                if morpheme_token_node_id <= node_id:
                    token_id = i + 1
                    token_node_id = node_id
                    break
            if token_node_id == 0:
                token_node_id = morpheme_token_node_id
                token_id = len(tokens) + 1
                tokens[token_node_id] = token
            del morpheme[5]
            morpheme[6] = token_id
            morpheme[7] = tokens[token_node_id]
            morpheme.append(is_gold)
            morpheme.insert(0, sent_id)
            lattice.append(morpheme)
    return pd.DataFrame(lattice, columns=column_names)


def _load_ud_conllu_partition(lattices_file_path, column_names):
    partition = []
    lattice_sentences = _split_sentences(lattices_file_path)
    for i, lattice in enumerate(lattice_sentences):
        sent_id = i + 1
        lattice = [line.replace("\t\t", "\t_\t").replace("\t\t", "\t_\t").split() for line in lattice if line[0] != '#']
        lattice = _normalize_lattice(lattice)
        df = _build_ud_sample(sent_id, lattice, column_names)
        partition.append(df)
    return partition


def _load_ud_conllu(tb_path, partition, column_names, lang, la_name, tb_name, ma_name, conll_type):
    treebank = {}
    for partition_type in partition:
        file_name = f'{la_name}_{tb_name}-ud-{partition_type}'.lower()
        if conll_type == 'conllul':
            lattices_path = tb_path / f'conllul/UL_{lang}-{tb_name}' / f'{file_name}.{ma_name}.{conll_type}'
        else:
            lattices_path = tb_path / f'UD_{lang}-{tb_name}' / f'{file_name}.{conll_type}'
        print(f'loading {lattices_path.stem} treebank file')
        lattices = _load_ud_conllu_partition(lattices_path, column_names)
        print(f'{partition_type} lattices: {len(lattices)}')
        treebank[partition_type] = lattices
    return treebank


def _load_gold_data_lattices(root_path, partition, la_name, tb_name, ma_name):
    try:
        gold_dataset = _load_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, partition, 'gold')
    except FileNotFoundError:
        gold_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, 'gold')
    return gold_dataset


def _save_gold(tb_path, root_path, partition, lang, la_name, tb_name, ma_name):
    gold_lattices = _load_ud_conllu(tb_path, partition, _lattice_fields, lang, la_name, tb_name, ma_name, 'conllu')
    gold_dataset = _to_data_lattices(gold_lattices)
    _save_data_lattices(root_path / la_name / tb_name, gold_dataset, 'gold')


def _save_uninfused_lattices(tb_path, root_path, partition, lang, la_name, tb_name, ma_name):
    lattices = _load_ud_conllu(tb_path, partition, _lattice_fields, lang, la_name, tb_name, ma_name, 'conllul')
    lattices_dataset = _to_data_lattices(lattices)
    gold_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, 'gold')
    valid_sent_mask = _validate_data_lattices(lattices_dataset, gold_dataset)
    if any([not any(valid_sent_mask[t]) for t in partition]):
        for partition_type in partition:
            lattices_dataset[partition_type] = [d for d, m in zip(lattices_dataset[partition_type], valid_sent_mask[partition_type]) if m]
            gold_dataset[partition_type] = [d for d, m in zip(gold_dataset[partition_type], valid_sent_mask[partition_type]) if m]
        _save_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, gold_dataset, 'gold')
        _save_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, lattices_dataset, 'uninf')
    else:
        _save_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, lattices_dataset, 'uninf')


def _save_infused_lattices(root_path, partition, la_name, tb_name, ma_name):
    lattices_dataset, gold_dataset = load_uninfused_lattices(root_path, partition, la_name, tb_name, ma_name)
    infused_lattices_dataset = infuse_tb_lattices(lattices_dataset, gold_dataset)
    _save_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, infused_lattices_dataset)


def _save_gold_token_super_tag(root_path, partition, la_name, tb_name):
    gold_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, 'gold')
    grouped_gold_dataset = get_grouped_analysis_dataset(gold_dataset, _lattice_fields)
    _save_data_lattices(root_path / la_name / tb_name / 'seq' / 'token-super-tag', grouped_gold_dataset, 'gold-super')


def load_gold(root_path, partition, la_name, tb_name):
    return _load_data_lattices(root_path / la_name / tb_name, partition, 'gold')


def load_gold_token_super_tag(root_path, partition, la_name, tb_name):
    return _load_data_lattices(root_path / la_name / tb_name / 'seq' / 'token-super-tag', partition, 'gold-super')


def load_infused_lattices(root_path, partition, la_name, tb_name, ma_name):
    gold_dataset = _load_gold_data_lattices(root_path, partition, la_name, tb_name, ma_name)
    infused_lattices_dataset = _load_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, partition)
    return infused_lattices_dataset, gold_dataset


def load_uninfused_lattices(root_path, partition, la_name, tb_name, ma_name):
    gold_dataset = _load_gold_data_lattices(root_path, partition, la_name, tb_name, ma_name)
    lattices_dataset = _load_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, partition, 'uninf')
    return lattices_dataset, gold_dataset


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/ud'
    tb_path = Path.home() / 'dev/onlplab/UniversalDependencies'
    langs = {'he': 'Hebrew', 'tr': 'Turkish'}
    tb_names = {'he': 'HTB', 'tr': 'IMST'}
    ma_names = {'he': 'heblex', 'tr': 'trmorph2'}
    for la_name in ['he', 'tr']:
        lang = langs[la_name]
        tb_name = tb_names[la_name]
        ma_name = ma_names[la_name]
        _save_gold(tb_path, root_path, partition, lang, la_name, tb_name, ma_name)
        _save_uninfused_lattices(tb_path, root_path, partition, lang, la_name, tb_name, ma_name)
        _save_infused_lattices(root_path, partition, la_name, tb_name, ma_name)
        _save_gold_token_super_tag(root_path, partition, la_name, tb_name)


if __name__ == '__main__':
    main()
