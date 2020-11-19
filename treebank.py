import unicodedata
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import os


_lattice_fields = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token', 'is_gold']


def _get_sent_indices_to_remove(dataset, tags):
    return [i + 1 for i, df in enumerate(dataset) if set(df.tag).intersection(tags)]


def _remove_from_partition(partition, tags):
    indices = _get_sent_indices_to_remove(partition[1], tags)
    lattices = [df for df in partition[0] if df.sent_id.unique() not in indices]
    gold_lattices = [df for df in partition[1] if df.sent_id.unique() not in indices]
    print(f'{len(indices)} removed samples')
    return lattices, gold_lattices


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
    if edge.to_node_id not in edges:
        analyses.append(analysis)
        return
    next_node = edges[edge.to_node_id]
    for i in range(len(next_node)):
        _dfs(edges, edge.to_node_id, i, analysis, analyses)


def _parse_sent_analyses(df, column_names):
    token_analyses = {}
    token_edges = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples():
        token_edges[row.token_id][row.from_node_id].append(row)
    for token_id in token_edges:
        analyses = []
        token_lattice_start_node_id = min(token_edges[token_id].keys())
        token_lattice_start_node = token_edges[token_id][token_lattice_start_node_id]
        for j in range(len(token_lattice_start_node)):
            _dfs(token_edges[token_id], token_lattice_start_node_id, j, [], analyses)
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


def _save_data_lattices(root_path, dataset, ner_feat, ner_only, baseline, data_type=None):
    os.makedirs(root_path, exist_ok=True)
    ner_suff = f'ner_{ner_feat}' if not ner_only else f'.ner_{ner_feat}_only'
    for partition_type in dataset:
        df = pd.concat(dataset[partition_type]).reset_index(drop=True)
        if data_type:
            file_path = root_path / f'{partition_type}-{baseline}-{data_type}.{ner_suff}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}-{baseline}.{ner_suff}.lattices.csv'

        df.to_csv(str(file_path))


def _load_data_lattices(root_path, partition, ner_feat, ner_only, baseline, data_type=None):
    dataset = {}
    ner_suff = f'ner_{ner_feat}' if not ner_only else f'.ner_{ner_feat}_only'
    for partition_type in partition:
        if data_type:
            file_path = root_path / f'{partition_type}-{baseline}-{data_type}.{ner_suff}.lattices.csv'
        else:
            file_path = root_path / f'{partition_type}-{baseline}.{ner_suff}.lattices.csv'
        print(f'loading {file_path}')
        df = pd.read_csv(str(file_path), index_col=0, keep_default_na=False)
        lattices = {sent_id: x.reset_index(drop=True) for sent_id, x in df.groupby(df.sent_id)}
        dataset[partition_type] = [lattices[sent_id] for sent_id in sorted(lattices)]
        print(f'{file_path.stem} data size: {len(dataset[partition_type])}')
    return dataset


def _to_data_lattices(treebank):
    dataset = {}
    column_names = _lattice_fields + ['analysis_id', 'morpheme_id']
    for partition_type in treebank:
        lattices = [_parse_sent_analyses(df, column_names) for df in treebank[partition_type]]
        dataset[partition_type] = lattices
    return dataset


def _assert_data(lattices, gold_lattices):
    for partition_type in gold_lattices:
        for df, gold_df in zip(lattices[partition_type], gold_lattices[partition_type]):
            assert df.sent_id.unique() == gold_df.sent_id.unique()
            assert len(df.groupby(df.token_id)) == len(gold_df.groupby(gold_df.token_id))


def _is_token_aligned(df, gold_df):
    gb = df.groupby(df.token_id)
    gold_gb = gold_df.groupby(gold_df.token_id)
    for token_id, token_df in sorted(gb):
        gold_token_df = gold_gb.get_group(token_id)
        token = token_df.token.unique().item()
        gold_token = gold_token_df.token.unique().item()
        if token != gold_token:
            return False
    return True


def _validate_data_lattices(lattices, gold_lattices):
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
            elif not _is_token_aligned(df, gold_df):
                print(f'sent {sent_id} misaligned tokens')
                align_mask[i] = False
        mask[partition_type] = align_mask
    return mask


# https://en.wikipedia.org/wiki/Unicode_character_property
# https://stackoverflow.com/questions/48496869/python3-remove-arabic-punctuation
def _normalize_unicode(s):
    return ''.join(c for c in s if not unicodedata.category(c).startswith('M'))


def _normalize_lattice(lattice):
    return [[_normalize_unicode(part) for part in morpheme] for morpheme in lattice]


# SPMRL Treebank
_spmrl_host_tags = ['COP', 'NN', 'RB', 'BN', 'IN', 'VB', 'NNT', 'NNP', 'PRP', 'QW', 'JJ', 'CD', 'POS', 'P', 'CC',
                    'BNT', 'AT', 'JJT', 'DTT', 'CDT', 'EX', 'DT', 'MD', 'INTJ', 'NEG', 'NCD']
_spmrl_pref_tags = ['CONJ', 'DEF', 'PREPOSITION', 'REL', 'ADVERB', 'TEMP']
_spmrl_suff_tags = ['S_PRN', 'S_ANP']


def _add_morpheme_type(dataset):
    print("Add morpheme type")
    total_tags = defaultdict(set)
    total_multi_host_tags = set()
    for partition_type in dataset:
        for df in dataset[partition_type]:
            morpheme_types, tags, multi_host_tags = _get_morpheme_types(df)
            total_tags.update(tags)
            total_multi_host_tags.update(multi_host_tags)
            df['morpheme_type'] = morpheme_types
    print(total_tags)
    print(total_multi_host_tags)


def _get_morpheme_types(df):
    morpheme_types = []
    tags = defaultdict(set)
    multi_host_tags = set()
    for (token_id, analysis_id), analysis_df in df.groupby([df.token_id, df.analysis_id]):
        prefixes, hosts, suffixes = [], [], []
        for m in analysis_df.itertuples():
            l = hosts
            m_type = 'host'
            if m.tag == 'yyQUOT' and len(analysis_df) > 1:
                m_type = 'pref'
                l = prefixes
            elif m.tag in _spmrl_suff_tags:
                m_type = 'suff'
                l = suffixes
            elif len(hosts) > 0:
                if m.tag == 'PRP' or m.tag == 'AT' or m.tag == 'DUMMY_AT':
                    m_type = 'suff'
                    l = suffixes
                else:
                    multi_host_tags.add(tuple(hosts))
                    # print(hosts)
                    # m_type = 'host'
                    # l = hosts
            elif m.tag in _spmrl_pref_tags:
                m_type = 'pref'
                l = prefixes
            morpheme_types.append(m_type)
            l.append(m.tag)
            tags[m_type].add(m.tag)
    return morpheme_types, tags, multi_host_tags


def _get_group_morpheme_type(df, lattice_fields):
    column_names = lattice_fields + ['analysis_id', 'morpheme_id', 'morpheme_type']
    sent_id = df.sent_id.unique().item()
    grouped_rows = []
    for (token_id, analysis_id), analysis_df in df.groupby([df.token_id, df.analysis_id]):
        token = analysis_df.token.unique().item()
        is_gold = analysis_df.is_gold.unique().item()
        type_morphemes = {}
        for morpheme_type, morpheme_type_df in analysis_df.groupby([analysis_df.morpheme_type]):
            morphemes = {m.morpheme_id: m for m in morpheme_type_df.itertuples()}
            from_ids = [morphemes[morpheme_id].from_node_id for morpheme_id in sorted(morphemes)]
            to_ids = [morphemes[morpheme_id].to_node_id for morpheme_id in sorted(morphemes)]
            forms = [morphemes[morpheme_id].form for morpheme_id in sorted(morphemes)]
            lemmas = [morphemes[morpheme_id].lemma for morpheme_id in sorted(morphemes)]
            tags = [morphemes[morpheme_id].tag for morpheme_id in sorted(morphemes)]
            feats = [morphemes[morpheme_id].feats for morpheme_id in sorted(morphemes)]
            row = [sent_id, from_ids[0], to_ids[-1], '-'.join(forms), '-'.join(lemmas), '-'.join(tags), '-'.join(feats),
                   token_id, token, is_gold, analysis_id]
            type_morphemes[morpheme_type] = row
        morpheme_id = 0
        for morpheme_type in ['pref', 'host', 'suff']:
            if morpheme_type in type_morphemes:
                row = type_morphemes[morpheme_type]
                row.append(morpheme_id)
                row.append(morpheme_type)
                grouped_rows.append(row)
                morpheme_id += 1
    return pd.DataFrame(grouped_rows, columns=column_names)


def _get_group_analysis(df, lattice_fields):
    column_names = lattice_fields + ['analysis_id', 'morpheme_id']
    sent_id = df.sent_id.unique().item()
    grouped_rows = []
    for (token_id, analysis_id), analysis_df in df.groupby([df.token_id, df.analysis_id]):
        token = str(analysis_df.token.unique().item())
        is_gold = analysis_df.is_gold.unique().item()
        morphemes = {m.morpheme_id: m for m in analysis_df.itertuples()}
        from_ids = [morphemes[morpheme_id].from_node_id for morpheme_id in sorted(morphemes)]
        to_ids = [morphemes[morpheme_id].to_node_id for morpheme_id in sorted(morphemes)]
        forms = [str(morphemes[morpheme_id].form) for morpheme_id in sorted(morphemes)]
        lemmas = [str(morphemes[morpheme_id].lemma) for morpheme_id in sorted(morphemes)]
        tags = [morphemes[morpheme_id].tag for morpheme_id in sorted(morphemes)]
        feats = [morphemes[morpheme_id].feats for morpheme_id in sorted(morphemes)]
        row = [sent_id, from_ids[0], to_ids[-1], '-'.join(forms), '-'.join(lemmas), '-'.join(tags), '-'.join(feats),
               token_id, token, is_gold, analysis_id, 0]
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows, columns=column_names)


def _get_grouped_morpheme_type_dataset(dataset, lattice_fields):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(_get_group_morpheme_type(df, lattice_fields))
    return grouped_dataset


def _get_grouped_analysis_dataset(dataset, lattice_fields):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(_get_group_analysis(df, lattice_fields))
    return grouped_dataset


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
        lattice = [line.split('\t') for line in lattice]
        df = _build_spmrl_sample(sent_id, lattice, tokens, column_names, is_gold)
        partition.append(df)
    return partition


def _load_spmrl_conll(tb_path, partition, column_names, lang, tb_name, ma_name=None):
    treebank = {}
    for partition_type in partition:
        file_name = f'{partition_type}_{tb_name}'.lower()
        print(f'loading {file_name} dataset')
        if ma_name is not None:
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.lattices'
        else:
            lattices_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}-gold.lattices'
        tokens_path = tb_path / f'{lang}Treebank' / tb_name / f'{file_name}.tokens'
        lattices = _load_spmrl_conll_partition(lattices_path, tokens_path, column_names, ma_name is None)
        print(f'{partition_type} lattices: {len(lattices)}')
        treebank[partition_type] = lattices
    return treebank


# UD Treebank
def _build_ud_sample(sent_id, ud_lattice, column_names):
    tokens = {}
    lattice = []
    # nodes = {0}
    for morpheme in ud_lattice:
        if len(morpheme[0].split('-')) == 2:
            from_node_id, to_node_id = (int(v) for v in morpheme[0].split('-'))
            # if from_node_id not in nodes:
            #     print(f'unreachable node id in morpheme row: {morpheme}')
            # nodes.add(to_node_id)
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
        # Bug fix - invalid lines (missing '_') found in the Hebrew treebank
        lattice = [line.replace("\t\t", "\t_\t").replace("\t\t", "\t_\t").split('\t') for line in lattice if line[0] != '#']
        # Bug fix - clean unicode characters
        lattice = _normalize_lattice(lattice)

        df = _build_ud_sample(sent_id, lattice, column_names)
        partition.append(df)
    return partition


def _load_ud_conllu(tb_path, partition, ner_feat, ner_only, baseline, column_names, lang, la_name, tb_name, ma_name=None):
    treebank = {}
    ner_suff = f'ner_{ner_feat}' if not ner_only else f'.ner_{ner_feat}_only'
    for partition_type in partition:
        if baseline == 'gold':
            file_name = f'{la_name}_{tb_name}-ud-{partition_type}'.lower()
        else:
            file_name = f'{la_name}_{tb_name}-ud-{partition_type}-{baseline}'.lower()
        if ma_name is not None:
            lattices_path = tb_path / f'conllul/UL_{lang}-{tb_name}-NER' / f'{file_name}.{ma_name}.{ner_suff}.conllul'
        else:
            lattices_path = tb_path / f'UD_{lang}-{tb_name}-NER' / f'{file_name}.{ner_suff}.conllu'
        print(f'loading {lattices_path} treebank file')
        lattices = _load_ud_conllu_partition(lattices_path, column_names)
        print(f'{partition_type} lattices: {len(lattices)}')
        treebank[partition_type] = lattices
    return treebank


# Save data lattices
def _get_infused_lattices(df, gold_df):
    mask1_columns = ['form', 'lemma', 'tag']
    mask2_columns = ['lemma', 'tag']
    mask3_columns = ['form', 'tag']
    infused_lattices = []
    token_gb = df.groupby(df.token_id)
    gold_token_gb = gold_df.groupby(gold_df.token_id)
    for token_id, analyses_df in token_gb:
        is_inf = [False] * len(analyses_df)
        analysis_gb = analyses_df.groupby(analyses_df.analysis_id)
        gold_ids = [analysis_id for analysis_id, analysis_df in analysis_gb if analysis_df.is_gold.to_numpy().all()]
        if len(gold_ids) == 0:
            gold_df = gold_token_gb.get_group(token_id)
            gold_feats = [set(f.split("|")) for f in gold_df.feats.values]
            max_gold_len = -1
            mask = mask1_columns
            while len(gold_ids) == 0:
                for analysis_id, analysis_df in analysis_gb:
                    if np.array_equal(analysis_df[mask].values, gold_df[mask].values):
                        feats = [set(f.split("|")) for f in analysis_df.feats.values]
                        feats_intersection = [f & g for f, g in zip(feats, gold_feats)]
                        if len(feats_intersection) > max_gold_len:
                            max_gold_len = len(feats_intersection)
                            gold_ids.append(analysis_id)
                if len(gold_ids) == 0 and mask == mask1_columns:
                    mask = mask2_columns
                elif len(gold_ids) == 0 and mask == mask2_columns:
                    mask = mask3_columns
                else:
                    break
        if len(gold_ids) > 1:
            raise Exception(f"multiple gold ids in {gold_df}")
        if len(gold_ids) == 0:
            print(f'infusing gold analysis: {gold_df.values.tolist()}')
            gold_id = analyses_df.analysis_id.max() + 1
            gold_df.analysis_id = gold_id
            analyses_df = pd.concat([analyses_df, gold_df], axis=0)
            is_inf.extend([True] * len(gold_df))
            gold_ids.append(gold_id)
        analyses_df['is_inf'] = is_inf
        analyses_df.is_gold = analyses_df.analysis_id == gold_ids[0]
        analyses_df.loc[analyses_df.analysis_id == gold_ids[0], 'is_gold'] = True
        infused_lattices.append(analyses_df)
    inf_df = pd.concat(infused_lattices).reset_index(drop=True)
    return inf_df


def _infuse_tb_lattices(lattices, base_lattices):
    dataset = {}
    for partition_type in lattices:
        dataset[partition_type] = [(df, gold_df) for df, gold_df in zip(lattices[partition_type],
                                                                        base_lattices[partition_type])]
    infused_dataset = defaultdict(list)
    for partition_type in dataset:
        print(f'infusing {partition_type} dataset')
        for (lattice_df, gold_df) in dataset[partition_type]:
            infused_dataset[partition_type].append(_get_infused_lattices(lattice_df, gold_df))
    return infused_dataset


def _save_base(tb_path, root_path, partition, ner_feat, ner_only, baseline, lang, la_name, tb_name, tb_scheme):
    if tb_scheme == 'SPMRL':
        remove_zvl = tb_name[-1] == 'z'
        base_lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name[:-1] if remove_zvl else tb_name)
        if remove_zvl:
            indices = _get_sent_indices_to_remove(base_lattices['train'], ['ZVL'])
            base_lattices['train'] = [df for df in base_lattices['train'] if df.sent_id.unique() not in indices]
    else:
        base_lattices = _load_ud_conllu(tb_path, partition, ner_feat, ner_only, baseline, _lattice_fields, lang, la_name, tb_name)
    base_dataset = _to_data_lattices(base_lattices)
    _save_data_lattices(root_path / la_name / f'{tb_name}-NER', base_dataset, ner_feat, ner_only, baseline)


def _save_base_morpheme_tag_type(root_path, partition, baseline, la_name, tb_name, mtag_level):
    base_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, baseline)
    _add_morpheme_type(base_dataset)
    _save_data_lattices(root_path / la_name / tb_name / 'seq' / f'{mtag_level}-mtag', base_dataset, baseline, 'type')


def _save_base_multi_tag(root_path, partition, baseline, la_name, tb_name, mtag_level):
    if mtag_level == 'token':
        base_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, baseline)
        grouped_dataset = _get_grouped_analysis_dataset(base_dataset, _lattice_fields)
    else:
        type_dataset = _load_data_lattices(root_path / la_name / tb_name / 'seq' / f'{mtag_level}-mtag', partition, baseline, 'type')
        grouped_dataset = _get_grouped_morpheme_type_dataset(type_dataset, _lattice_fields)
    _save_data_lattices(root_path / la_name / tb_name / 'seq' / f'{mtag_level}-mtag', grouped_dataset, baseline, 'mtag')


def _save_uninfused_lattices(tb_path, root_path, partition, ner_feat, ner_only, baseline, lang, la_name, tb_name, ma_name, tb_scheme):
    base_dataset = _load_data_lattices(root_path / la_name / f'{tb_name}-NER', partition, ner_feat, ner_only, baseline)
    if tb_scheme == 'SPMRL':
        remove_zvl = tb_name[-1] == 'z'
        lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name[:-1] if remove_zvl else tb_name, ma_name)
        if remove_zvl:
            base_lattices = _load_spmrl_conll(tb_path, partition, _lattice_fields, lang, tb_name[:-1] if remove_zvl else tb_name)
            indices = _get_sent_indices_to_remove(base_lattices['train'], ['ZVL'])
            lattices['train'] = [df for df in lattices['train'] if df.sent_id.unique() not in indices]
    else:
        lattices = _load_ud_conllu(tb_path, partition, ner_feat, ner_only, baseline, _lattice_fields, lang, la_name, tb_name, ma_name)
    lattices_dataset = _to_data_lattices(lattices)
    valid_sent_mask = _validate_data_lattices(lattices_dataset, base_dataset)
    if any([not all(valid_sent_mask[t]) for t in partition]):
        for partition_type in partition:
            lattices_dataset[partition_type] = [d for d, m in zip(lattices_dataset[partition_type], valid_sent_mask[partition_type]) if m]
            base_dataset[partition_type] = [d for d, m in zip(base_dataset[partition_type], valid_sent_mask[partition_type]) if m]
        _save_data_lattices(root_path / la_name / f'{tb_name}-NER' / 'lattice' / ma_name, base_dataset, baseline)
        _save_data_lattices(root_path / la_name / f'{tb_name}-NER' / 'lattice' / ma_name, lattices_dataset, baseline, 'uninf')
    else:
        _save_data_lattices(root_path / la_name / f'{tb_name}-NER' / 'lattice' / ma_name, lattices_dataset, baseline, 'uninf')


def _save_infused_lattices(root_path, partition, ner_feat, ner_only, baseline, la_name, tb_name, ma_name):
    lattices_dataset, base_dataset = tb_load_lattices(root_path, partition, baseline, la_name, tb_name, ma_name, 'uninf')
    infused_lattices_dataset = _infuse_tb_lattices(lattices_dataset, base_dataset)
    _save_data_lattices(root_path / la_name / f'{tb_name}-NER' / 'lattice' / ma_name, infused_lattices_dataset, baseline, 'inf')


# Load data lattices
def _load_base_data_lattices(root_path, partition, baseline, la_name, tb_name, ma_name):
    try:
        base_dataset = _load_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, partition, baseline)
    except FileNotFoundError:
        base_dataset = _load_data_lattices(root_path / la_name / tb_name, partition, baseline)
    return base_dataset


# API ##################################################################################################################
def tb_load_base(root_path, partition, baseline, la_name, tb_name):
    return _load_data_lattices(root_path / la_name / tb_name, partition, baseline)


def tb_load_base_mtag(root_path, partition, baseline, la_name, tb_name, mtag_level):
    return _load_data_lattices(root_path / la_name / tb_name / 'seq' / f'{mtag_level}', partition, baseline, 'mtag')


def tb_load_lattices(root_path, partition, baseline, la_name, tb_name, ma_name, inf_type):
    base_dataset = _load_base_data_lattices(root_path, partition, baseline, la_name, tb_name, ma_name)
    lattices_dataset = _load_data_lattices(root_path / la_name / tb_name / 'lattice' / ma_name, partition, baseline, inf_type)
    return lattices_dataset, base_dataset


def tb_export_tokens(root_path, tb_path, partition, ner_feat, ner_only, lang, la_name, tb_name):
    ner_suff = f'ner_{ner_feat}' if not ner_only else f'.ner_{ner_feat}_only'
    dataset = tb_load_base(root_path, partition, 'gold', la_name, tb_name)
    for partition_type in dataset:
        tokens = []
        for x in dataset[partition_type]:
            d = {int(j): str(y.token.unique().item()) for j, y in x.groupby(x.token_id)}
            sent_tokens = [d[i] for i in sorted(d)]
            tokens.append(sent_tokens)
        file_name = f'{la_name}_{tb_name}-ud-{partition_type}'.lower()
        file_path = tb_path / f'UD_{lang}-{tb_name}-NER' / f'{file_name}.{ner_suff}.tokens.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            for sent_tokens in tokens:
                f.write('\n'.join(sent_tokens))
                f.write('\n\n')
            f.write('\n')
# API ##################################################################################################################


def main():
    scheme = 'UD'
    partition = ['dev', 'test', 'train']
    ner_feat = ['plo', 'nocat', 'full']
    root_path = Path.home() / f'dev/aseker00/modi/tb/{scheme}'
    tb_path = Path.home() / f'dev/aseker00/modi/data/for_amit'
    la_name = 'he'
    ma_name = 'heblex'
    lang = 'Hebrew'
    tb_name = 'HTB'
    # tb_export_tokens(root_path, tb_path, partition, ner_feat[0], False, lang, la_name, tb_name)
    # _save_base(tb_path, root_path, partition, ner_feat[0], False, 'gold', lang, la_name, tb_name, scheme)
    _save_uninfused_lattices(tb_path, root_path, partition[:1], ner_feat[0], False, 'gold', lang, la_name, tb_name, ma_name, scheme)
    _save_infused_lattices(root_path, partition[:1], ner_feat[0], False, 'gold', la_name, tb_name, ma_name)


if __name__ == '__main__':
    main()
