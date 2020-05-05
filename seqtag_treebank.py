from collections import defaultdict
import pandas as pd


host_tags = ['COP', 'NN', 'RB', 'BN', 'IN', 'VB', 'NNT', 'NNP', 'PRP', 'QW', 'JJ', 'CD', 'POS', 'P', 'CC',  'BNT',
             'AT', 'JJT', 'DTT', 'CDT', 'EX', 'DT', 'MD', 'INTJ', 'NEG', 'NCD']
pref_tags = ['CONJ', 'DEF', 'PREPOSITION', 'REL', 'ADVERB', 'TEMP']
suff_tags = ['S_PRN', 'S_ANP']


def get_morpheme_types(df):
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
            elif m.tag in suff_tags:
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
            elif m.tag in pref_tags:
                m_type = 'pref'
                l = prefixes
            morpheme_types.append(m_type)
            l.append(m.tag)
            tags[m_type].add(m.tag)
    return morpheme_types, tags, multi_host_tags


def get_group_morpheme_type(df, lattice_fields):
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


def get_group_analysis(df, lattice_fields):
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


def add_morpheme_type(dataset):
    print("Add morpheme type")
    total_tags = defaultdict(set)
    total_multi_host_tags = set()
    for partition_type in dataset:
        for df in dataset[partition_type]:
            morpheme_types, tags, multi_host_tags = get_morpheme_types(df)
            total_tags.update(tags)
            total_multi_host_tags.update(multi_host_tags)
            df['morpheme_type'] = morpheme_types
    print(total_tags)
    print(total_multi_host_tags)


def get_grouped_morpheme_type_dataset(dataset, lattice_fields):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(get_group_morpheme_type(df, lattice_fields))
    return grouped_dataset


def get_grouped_analysis_dataset(dataset, lattice_fields):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(get_group_analysis(df, lattice_fields))
    return grouped_dataset
