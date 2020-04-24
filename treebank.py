from collections import defaultdict
from copy import deepcopy
import pandas as pd


def split_sentences(file_path):
    with open(str(file_path)) as f:
        lines = [line.strip() for line in f.readlines()]
    sent_sep_pos = [i for i in range(len(lines)) if len(lines[i]) == 0]
    sent_sep = [(0, sent_sep_pos[0])] + [(sent_sep_pos[i]+1, sent_sep_pos[i+1]) for i in range(len(sent_sep_pos) - 1)]
    return [lines[sep[0]:sep[1]] for sep in sent_sep]


def build_sample_df(sent_id, lattice, tokens, column_names):
    for morpheme in lattice:
        morpheme[0] = int(morpheme[0])
        morpheme[1] = int(morpheme[1])
        morpheme[7] = int(morpheme[7])
        morpheme.append(tokens[morpheme[7]])
        morpheme.insert(0, sent_id)
        del morpheme[6]
    return pd.DataFrame(lattice, columns=column_names)


def load_treebank_partition(lattices_file_path, gold_file_path, tokens_file_path, column_names):
    lattices_partition = []
    gold_lattices_partition = []
    lattice_sentences = split_sentences(lattices_file_path)
    gold_sentences = split_sentences(gold_file_path)
    token_sentences = split_sentences(tokens_file_path)
    for i, (lattice, gold, tokens) in enumerate(zip(lattice_sentences, gold_sentences, token_sentences)):
        sent_id = i + 1
        tokens = {j + 1: t for j, t in enumerate(tokens)}
        lattice = [line.split() for line in lattice]
        gold = [line.split() for line in gold]
        df = build_sample_df(sent_id, lattice, tokens, column_names)
        gold_df = build_sample_df(sent_id, gold, tokens, column_names)
        lattices_partition.append(df)
        gold_lattices_partition.append(gold_df)
    return lattices_partition, gold_lattices_partition


def load_treebank(tb_path, partition):
    tb = {}
    column_names = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token']
    for partition_type in partition:
        print(f'loading {partition_type} treebank dataset')
        lattices_path = tb_path / f'{partition_type}_hebtb.lattices'
        gold_lattices_path = tb_path / f'{partition_type}_hebtb-gold.lattices'
        tokens_path = tb_path / f'{partition_type}_hebtb.tokens'
        tb[partition_type] = load_treebank_partition(lattices_path, gold_lattices_path, tokens_path, column_names)
        print(f'{partition_type} treebank lattices: {len(tb[partition_type][0])}')
        print(f'{partition_type} treebank gold lattices: {len(tb[partition_type][1])}')
    return tb


def dfs(edges, cur_node_id, next_node_id, analysis_in, analyses):
    node = edges[cur_node_id]
    edge = node[next_node_id]
    analysis = deepcopy(analysis_in)
    analysis.append(tuple(edge))
    if edge.to_node_id not in edges:
        analyses.append(analysis)
        return
    next_node = edges[edge.to_node_id]
    for i in range(len(next_node)):
        dfs(edges, edge.to_node_id, i, analysis, analyses)


def parse_sent_analyses(df, column_names):
    token_analyses = {}
    edges = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples():
        edges[row.token_id][row.from_node_id].append(row)
    for token_id in edges:
        analyses = []
        token_lattice_start_node_id = min(edges[token_id].keys())
        token_lattice_start_node = edges[token_id][token_lattice_start_node_id]
        for j in range(len(token_lattice_start_node)):
            dfs(edges[token_id], token_lattice_start_node_id, j, [], analyses)
        token_analyses[token_id] = analyses
    return lattice_to_dataframe(token_analyses, column_names)


def lattice_to_dataframe(lattice, column_names):
    rows = []
    for token_id in lattice:
        for i, analyses in enumerate(lattice[token_id]):
            for j, morpheme in enumerate(analyses):
                row = [*morpheme[1:], i, j]
                rows.append(row)
    # return pd.merge(lattice_df, token_df, on='token_id')
    return pd.DataFrame(rows, columns=column_names)


def get_sent_indices_to_remove(dataset, tags):
    return [i + 1 for i, df in enumerate(dataset) if set(df.tag).intersection(tags)]


def get_dataset(tb):
    lattices_dataset = {}
    gold_lattices_dataset = {}
    column_names = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token']
    column_names += ['analysis_id', 'morpheme_id']
    for partition_type in tb:
        if partition_type == 'train':
            indices = get_sent_indices_to_remove(tb[partition_type][1], ['ZVL'])
            lattices = [df for df in tb[partition_type][0] if df.sent_id.unique() not in indices]
            gold_lattices = [df for df in tb[partition_type][1] if df.sent_id.unique() not in indices]
            tb[partition_type] = (lattices, gold_lattices)
            print(f'{len(indices)} removed samples')
        lattices = [parse_sent_analyses(df, column_names) for df in tb[partition_type][0]]
        gold_lattices = [parse_sent_analyses(df, column_names) for df in tb[partition_type][1]]
        print(f'{partition_type} lattices: {len(lattices)}')
        print(f'{partition_type} gold lattices: {len(gold_lattices)}')
        lattices_dataset[partition_type] = lattices
        gold_lattices_dataset[partition_type] = gold_lattices
    return lattices_dataset, gold_lattices_dataset


def assert_dataset(lattices, gold_lattices):
    for partition_type in gold_lattices:
        for df, gold_df in zip(lattices[partition_type], gold_lattices[partition_type]):
            assert df.sent_id.unique() == gold_df.sent_id.unique()
            assert len(df.groupby(df.token_id)) == len(gold_df.groupby(gold_df.token_id))


def save_lattices_dataset(root_path, dataset, suffix):
    for partition_type in dataset:
        df = pd.concat(dataset[partition_type]).reset_index(drop=True)
        file_path = root_path / f'{partition_type}-{suffix}.csv'
        df.to_csv(str(file_path))


def load_lattices_dataset(root_path, partition, suffix):
    dataset = {}
    for partition_type in partition:
        print(f'loading {partition_type}-{suffix} dataset')
        file_path = root_path / f'{partition_type}-{suffix}.csv'
        df = pd.read_csv(str(file_path), index_col=0)
        lattices = {sent_id: x.reset_index(drop=True) for sent_id, x in df.groupby(df.sent_id)}
        dataset[partition_type] = [lattices[sent_id] for sent_id in sorted(lattices)]
        print(f'{partition_type}-{suffix} data size: {len(dataset[partition_type])}')
    return dataset
