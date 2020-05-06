from collections import defaultdict
import pandas as pd
import numpy as np


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


def infuse_tb_lattices(lattices, gold_lattices):
    dataset = {}
    for partition_type in lattices:
        dataset[partition_type] = [(df, gold_df) for df, gold_df in zip(lattices[partition_type],
                                                                        gold_lattices[partition_type])]
    infused_dataset = defaultdict(list)
    for partition_type in dataset:
        print(f'infusing {partition_type} dataset')
        for (lattice_df, gold_df) in dataset[partition_type]:
            infused_dataset[partition_type].append(_get_infused_lattices(lattice_df, gold_df))
    return infused_dataset
