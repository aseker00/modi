from treebank import *
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


def parse_feats(feats_str):
    feats = defaultdict(list)
    kv = [kv.split('=') for kv in feats_str.split('|') if kv != '_']
    for [k, v] in kv:
        feats[k].append(v)
    return feats


def get_feat_names(dataset):
    data_feats = [parse_feats(f) for partition_type in dataset for df in dataset[partition_type] for f in df.feats]
    return set([k for d in data_feats for k in d.keys()])


def insert_df_feat_columns(df, vocab_feat_names):
    df_feats = [parse_feats(f) for f in df.feats]
    df_split_feats = [{f'feat_{key}': ''.join(feats.get(key, ['_'])) for key in vocab_feat_names} for feats in df_feats]
    return pd.concat([df, pd.DataFrame(df_split_feats).reset_index(drop=True)], axis=1)
    # return pd.concat([df.iloc[:, :df_feats_column_index + 1], pd.DataFrame(df_split_feats),
    #                   df.iloc[:, df_feats_column_index + 1:]], axis=1)


def get_dataset_with_feat_columns(dataset, vocab_feat_names):
    new_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            new_df = insert_df_feat_columns(df, vocab_feat_names)
            new_dataset[partition_type].append(new_df)
    return new_dataset


def match_gold_analysis(token_df, gold_token_analysis_df, mask):
    is_gold = [False] * token_df.analysis_id.unique().size
    gold_analysis = gold_token_analysis_df.where(mask).dropna(axis=1)
    for analysis_id, analysis_df in token_df.groupby(token_df.analysis_id):
        if len(analysis_df) != len(gold_token_analysis_df):
            continue
        analysis = analysis_df.where(mask).dropna(axis=1)
        is_gold[analysis_id] = np.array_equal(analysis.values, gold_analysis.values)
    return is_gold


def infuse_gold_token(analyses, gold_analysis, gold_mask):
    gold_infused = [False] * len(gold_mask)
    gold_duplicated = [False] * len(gold_mask)
    if sum(gold_mask) == 0:
        inf_id = analyses.analysis_id.max() + 1
        gold_analysis.analysis_id = inf_id
        analyses = pd.concat([analyses, gold_analysis], axis=0)
        gold_mask.append(True)
        gold_infused.append(True)
        gold_duplicated.append(False)
    elif sum(gold_mask) > 1:
        for i in [i for i, m in enumerate(gold_mask) if m][1:]:
            gold_duplicated[i] = True
    for i in range(len(gold_mask)):
        is_gold = gold_mask[i]
        is_inf = gold_infused[i]
        is_dup = gold_duplicated[i]
        analyses.loc[analyses.analysis_id == i, 'is_gold'] = is_gold
        analyses.loc[analyses.analysis_id == i, 'is_inf'] = is_inf
        analyses.loc[analyses.analysis_id == i, 'is_dup'] = is_dup
    return analyses


def infuse_gold(df, gold_df, feats):
    column_mask = [n in ['form', 'lemma', 'tag'] + feats for n in list(df)]
    column_mask_1 = [n in ['lemma', 'tag'] + feats for n in list(df)]
    column_mask_2 = [n in ['form', 'tag'] + feats for n in list(df)]
    gb = df.groupby(df.token_id)
    gold_gb = gold_df.groupby(gold_df.token_id)
    token_groups = {k: v for k, v in gb}
    gold_token_groups = {k: v for k, v in gold_gb}
    infused = []
    for token_id in sorted(token_groups):
        token_df = token_groups[token_id]
        gold_token_analysis_df = gold_token_groups[token_id]
        gold_token_analysis_mask = [[m1 & m2 for m1, m2 in zip(m, column_mask)] for m in
                                    (gold_token_analysis_df != '_').values.tolist()]
        gold_token_analysis_match_mask = match_gold_analysis(token_df, gold_token_analysis_df, gold_token_analysis_mask)
        if sum(gold_token_analysis_match_mask) == 0:
            gold_token_analysis_mask_1 = [[m1 & m2 for m1, m2 in zip(m, column_mask_1)] for m in
                                          (gold_token_analysis_df != '_').values.tolist()]
            gold_token_analysis_match_mask = match_gold_analysis(token_df, gold_token_analysis_df,
                                                                 gold_token_analysis_mask_1)
        if sum(gold_token_analysis_match_mask) == 0:
            gold_token_analysis_mask_2 = [[m1 & m2 for m1, m2 in zip(m, column_mask_2)] for m in
                                          (gold_token_analysis_df != '_').values.tolist()]
            gold_token_analysis_match_mask = match_gold_analysis(token_df, gold_token_analysis_df,
                                                                 gold_token_analysis_mask_2)
        if sum(gold_token_analysis_match_mask) == 0:
            print(f'infused analysis: {gold_token_analysis_df.values.tolist()}')
        infused_df = infuse_gold_token(token_df, gold_token_analysis_df, gold_token_analysis_match_mask)
        infused.append(infused_df)
    return pd.concat(infused)


def infuse_dataset(lattices, gold_lattices, feats):
    dataset = {}
    for partition_type in lattices:
        dataset[partition_type] = [(df, gold_df) for df, gold_df in zip(lattices[partition_type],
                                                                        gold_lattices[partition_type])]
    infused = defaultdict(list)
    for partition_type in dataset:
        print(f'infusing {partition_type} dataset')
        for (lattice_df, gold_df) in dataset[partition_type]:
            infused[partition_type].append(infuse_gold(lattice_df, gold_df, feats))
    return infused


def save_treebank_dataset(tb_path, root_path, partition):
    tb_dataset = load_treebank(tb_path, partition)
    lattices, gold_lattices = get_dataset(tb_dataset)
    save_lattices_dataset(root_path / 'lattice', lattices, 'lattices')
    save_lattices_dataset(root_path / 'lattice', gold_lattices, 'gold-lattices')


def save_dataset_with_feats(root_path, partition):
    lattices = load_lattices_dataset(root_path / 'lattice', partition, 'lattices')
    gold_lattices = load_lattices_dataset(root_path / 'lattice', partition, 'gold-lattices')
    vocab_feat_names = get_feat_names(lattices)
    feat_lattices = get_dataset_with_feat_columns(lattices, vocab_feat_names)
    feat_gold_lattices = get_dataset_with_feat_columns(gold_lattices, vocab_feat_names)
    save_lattices_dataset(root_path / 'lattice', feat_lattices, 'lattices-feats')
    save_lattices_dataset(root_path / 'lattice', feat_gold_lattices, 'gold-lattices-feats')


def save_infused_dataset(root_path, partition):
    lattices = load_lattices_dataset(root_path / 'lattice', partition, 'lattices-feats')
    gold_lattices = load_lattices_dataset(root_path / 'lattice', partition, 'gold-lattices-feats')
    feats = [f'feat_{f}' for f in ['gen', 'num', 'per', 'suf_gen', 'suf_num', 'suf_per']]
    infused_lattices = infuse_dataset(lattices, gold_lattices, feats)
    save_lattices_dataset(root_path / 'lattice', infused_lattices, 'lattices-inf')


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
    tb_path = Path.home() / 'dev/onlplab/HebrewResources/HebrewTreebank/hebtb'
    save_treebank_dataset(tb_path, root_path, partition)
    save_dataset_with_feats(root_path, partition)
    save_infused_dataset(root_path, partition)


if __name__ == '__main__':
    main()
