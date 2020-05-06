from collections import defaultdict
from copy import deepcopy
import pandas as pd


lattice_fields = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token',
                  'is_gold']


def split_sentences(file_path):
    with open(str(file_path)) as f:
        lines = [line.strip() for line in f.readlines()]
    sent_sep_pos = [i for i in range(len(lines)) if len(lines[i]) == 0]
    sent_sep = [(0, sent_sep_pos[0])] + [(sent_sep_pos[i]+1, sent_sep_pos[i+1]) for i in range(len(sent_sep_pos) - 1)]
    return [lines[sep[0]:sep[1]] for sep in sent_sep]


def _dfs(edges, cur_node_id, next_node_id, analysis_in, analyses):
    node = edges[cur_node_id]
    edge = node[next_node_id]
    analysis = deepcopy(analysis_in)
    analysis.append(tuple(edge))
    if edge.to_node_id not in edges:
        analyses.append(analysis)
        return
    next_node = edges[edge.to_node_id]
    for i in range(len(next_node)):
        _dfs(edges, edge.to_node_id, i, analysis, analyses)


def _parse_sent_analyses(df, column_names):
    token_analyses = {}
    edges = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples():
        edges[row.token_id][row.from_node_id].append(row)
    for token_id in edges:
        analyses = []
        token_lattice_start_node_id = min(edges[token_id].keys())
        token_lattice_start_node = edges[token_id][token_lattice_start_node_id]
        for j in range(len(token_lattice_start_node)):
            _dfs(edges[token_id], token_lattice_start_node_id, j, [], analyses)
        token_analyses[token_id] = analyses
    return _lattice_to_dataframe(token_analyses, column_names)


def _lattice_to_dataframe(lattice, column_names):
    rows = []
    for token_id in lattice:
        for i, analyses in enumerate(lattice[token_id]):
            for j, morpheme in enumerate(analyses):
                row = [*morpheme[1:], i, j]
                rows.append(row)
    # return pd.merge(lattice_df, token_df, on='token_id')
    return pd.DataFrame(rows, columns=column_names)


def save_tb_lattice_data(root_path, dataset, data_type=None):
    for partition_type in dataset:
        df = pd.concat(dataset[partition_type]).reset_index(drop=True)
        if data_type is not None:
            file_path = root_path / f'{partition_type}-{data_type}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}.lattices.csv'
        df.to_csv(str(file_path))


def load_tb_lattice_data(root_path, partition, data_type=None):
    dataset = {}
    for partition_type in partition:
        if data_type is not None:
            file_path = root_path / f'{partition_type}-{data_type}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}.lattices.csv'
        print(f'loading {file_path}')
        df = pd.read_csv(str(file_path), index_col=0)
        lattices = {sent_id: x.reset_index(drop=True) for sent_id, x in df.groupby(df.sent_id)}
        dataset[partition_type] = [lattices[sent_id] for sent_id in sorted(lattices)]
        print(f'{file_path} data size: {len(dataset[partition_type])}')
    return dataset


def get_tb_data(treebank):
    dataset = {}
    column_names = lattice_fields + ['analysis_id', 'morpheme_id']
    for partition_type in treebank:
        lattices = [_parse_sent_analyses(df, column_names) for df in treebank[partition_type]]
        dataset[partition_type] = lattices
    return dataset


def assert_treebank(lattices, gold_lattices):
    for partition_type in gold_lattices:
        for df, gold_df in zip(lattices[partition_type], gold_lattices[partition_type]):
            assert df.sent_id.unique() == gold_df.sent_id.unique()
            assert len(df.groupby(df.token_id)) == len(gold_df.groupby(gold_df.token_id))


def is_token_aligned(df, gold_df):
    gb = df.groupby(df.token_id)
    gold_gb = gold_df.groupby(gold_df.token_id)
    for token_id, token_df in sorted(gb):
        gold_token_df = gold_gb.get_group(token_id)
        token = token_df.token.unique().item()
        gold_token = gold_token_df.token.unique().item()
        if token != gold_token:
            return False
    return True


def validate_lattices(lattices, gold_lattices):
    mask = {}
    for partition_type in lattices:
        align_mask = [True] * len(lattices[partition_type])
        for i, (df, gold_df) in enumerate(zip(lattices[partition_type], gold_lattices[partition_type])):
            sent_id = df.sent_id.unique()
            if df.sent_id.unique() != gold_df.sent_id.unique():
                print(f'sent {sent_id} sent id mismatch')
                align_mask[i] = False
            elif len(df.groupby(df.token_id)) != len(gold_df.groupby(gold_df.token_id)):
                print(f'sent {sent_id} token num mismatch')
                align_mask[i] = False
            elif not is_token_aligned(df, gold_df):
                print(f'sent {sent_id} misaligned tokens')
                align_mask[i] = False
        mask[partition_type] = align_mask
    return mask
