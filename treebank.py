import pickle
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import pandas as pd


def _split_sentences(file_path):
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
    if edge[1] not in edges:
        analyses.append(analysis)
        return
    next_node = edges[edge[1]]
    for i in range(len(next_node)):
        _dfs(edges, edge[1], i, analysis, analyses)


def _parse_lattices(lattices_file_path, remove_zvl):
    sentences = _split_sentences(lattices_file_path)
    lattices = []
    removed_idx = []
    for i, sent_lattice in enumerate(sentences):
        data = [[int(value) if i in [0, 1, 7] else value for i, value in enumerate(line.split())]
                for line in sent_lattice]
        if remove_zvl and any([row[4] == 'ZVL' for row in data]):
            print(f'filter ZVL: {i}: {data}')
            removed_idx.append(i)
            continue
        x = [len(d) for d in data]
        if not x.count(x[0]) == len(x):
            print(f'filter FORMAT: {i}: {data}')
            continue
        lattice_df = pd.DataFrame(data, columns=_lattice_edge_keys)
        # merged_df = pd.merge(lattice_df, token_df, on='token_id')
        token_analyses = {}
        edges = defaultdict(lambda: defaultdict(list))
        for row in lattice_df.itertuples():
            edges[row[-1]][row[1]].append(row[1:])
        for token_id in edges:
            analyses = []
            token_lattice_start_node_id = min(edges[token_id].keys())
            token_lattice_start_node = edges[token_id][token_lattice_start_node_id]
            for i in range(len(token_lattice_start_node)):
                _dfs(edges[token_id], token_lattice_start_node_id, i, [], analyses)
            token_analyses[token_id] = analyses
        lattices.append(token_analyses)
    return lattices, removed_idx


def _transform_lattices_to_dataframe(lattices, tokens):
    print("Transform lattices to dataframe")
    rows = []
    for i, (sent_lattice, sent_tokens) in enumerate(zip(lattices, tokens)):
        sent_id = i + 1
        for j in range(len(sent_tokens)):
            token = sent_tokens[j]
            token_id = j + 1
            token_lattice = sent_lattice[token_id]
            for k, analysis in enumerate(token_lattice):
                for l, morpheme in enumerate(analysis):
                    rows.append([sent_id, token_id, k, l,
                                 morpheme[0], morpheme[1], morpheme[2], morpheme[3], morpheme[4], morpheme[6], token])
    return pd.DataFrame(rows, columns=_dataframe_columns)


def _identify_morpheme_host_and_affixes(df):
    analysis_gb = df.groupby(['sent_id', 'token_id', 'analysis_id'])
    morpheme_types = []
    tags = defaultdict(set)
    for x in analysis_gb.groups:
        g = analysis_gb.get_group(x)
        prefixes, hosts, suffixes = [], [], []
        for i in range(len(g)):
            m = g.iloc[i]
            if m.tag == 'yyQUOT' and len(g) > 1:
                m_type = 'pref'
                l = prefixes
            elif m.tag in suff_tags:
                m_type = 'suff'
                l = suffixes
            elif len(hosts) > 0:
                if m.tag == 'PRP' or m.tag == 'AT' or m.tag == 'DUMMY_AT':
                    m_type = 'suff'
                    l = suffixes
                else:
                    print(hosts)
                    m_type = 'host'
                    l = hosts
            elif m.tag in pref_tags:
                m_type = 'pref'
                l = prefixes
            else:
                m_type = 'host'
                l = hosts
            morpheme_types.append(m_type)
            l.append(m.tag)
            tags[m_type].add(m.tag)
    print(tags)
    return morpheme_types


def _add_morpheme_type(df):
    print("Add morpheme type")
    morpheme_types = _identify_morpheme_host_and_affixes(df)
    df.insert(5, 'morpheme_type', morpheme_types)
    return df


def _process_treebank(tb_root_dir_path, tb_type):
    token_file_path = tb_root_dir_path / f'{tb_type}_hebtb.tokens'
    gold_lattices_file_path = tb_root_dir_path / f'{tb_type}_hebtb-gold.lattices'
    tokens = _split_sentences(token_file_path)
    gold_lattices, removed_idx = _parse_lattices(gold_lattices_file_path, tb_type=='train')
    if removed_idx:
        tokens = [t for i, t in enumerate(tokens) if i not in removed_idx]
    gold_df = _transform_lattices_to_dataframe(gold_lattices, tokens)
    return gold_df


def dataframe_to_lattices(df):
    print("Transform dataframe into lattices")
    lattices = {}
    for row in df.itertuples():
        if row.sent_id not in lattices:
            lattices[row.sent_id] = {}
        if row.token_id not in lattices[row.sent_id]:
            lattices[row.sent_id][row.token_id] = defaultdict(list)
        lattices[row.sent_id][row.token_id][row.analysis_id].append(row[5:])
    return lattices


host_tags = ['COP', 'NN', 'RB', 'BN', 'IN', 'VB', 'NNT', 'NNP', 'PRP', 'QW', 'JJ', 'CD', 'POS', 'P', 'CC',  'BNT',
             'AT', 'JJT', 'DTT', 'CDT', 'EX', 'DT', 'MD', 'INTJ', 'NEG', 'NCD']
pref_tags = ['CONJ', 'DEF', 'PREPOSITION', 'REL', 'ADVERB', 'TEMP']
suff_tags = ['S_PRN', 'S_ANP']
_dataframe_columns = ['sent_id', 'token_id', 'analysis_id', 'morpheme_id', 'from_node_id', 'to_node_id', 'form',
                      'lemma', 'tag', 'feats', 'token']
_lattice_edge_keys = ['from_id', 'to_id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'token_id']


def _load_dataframe(data_dir_path, partition_type, lattices_type):
    csv_file_path = data_dir_path / f'{partition_type}-{lattices_type}.csv'
    return pd.read_csv(str(csv_file_path))


def load_dataframe(data_dir, partition):
    return {partition_type: _load_dataframe(data_dir, partition_type, 'gold-lattices-with-type')
            for partition_type in partition}


def load_lattices(root_dir_path, partition):
    dataframes = load_dataframe(root_dir_path / 'data', partition)
    return {partition_type: dataframe_to_lattices(dataframes[partition_type]) for partition_type in dataframes}


def main():
    tb_root_dir_path = Path.home() / 'dev/onlplab/HebrewResources/HebrewTreebank/hebtb'
    root_dir_path = Path.home() / 'dev/aseker00/modi'
    partition = ['dev', 'test', 'train']
    for partition_type in partition:
        gold_df = _process_treebank(tb_root_dir_path, partition_type)
        gold_lattices_file_path = root_dir_path / 'data' / f'{partition_type}-gold-lattices.csv'
        gold_df.to_csv(str(gold_lattices_file_path))
    for partition_type in partition:
        gold_lattices_file_path = root_dir_path / f'{partition_type}-gold-lattices'
        gold_lattices_with_type_file_path = root_dir_path / 'data' / f'{partition_type}-gold-lattices-with-type.csv'
        gold_df = pd.read_csv(gold_lattices_file_path)
        gold_df = _add_morpheme_type(gold_df)
        gold_df.to_csv(gold_lattices_with_type_file_path)


if __name__ == '__main__':
    main()
