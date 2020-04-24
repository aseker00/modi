from treebank import *
from collections import defaultdict
from pathlib import Path


def identify_morpheme_host_and_affixes(df):
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


def get_group_morpheme_type(df):
    column_names = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token',
                    'analysis_id', 'morpheme_id', 'morpheme_type']
    sent_id = df.sent_id.unique().item()
    grouped_rows = []
    for (token_id, analysis_id), analysis_df in df.groupby([df.token_id, df.analysis_id]):
        token = analysis_df.token.unique().item()
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
                   token_id, token, analysis_id]
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


def get_group_analysis(df):
    column_names = ['sent_id', 'from_node_id', 'to_node_id', 'form', 'lemma', 'tag', 'feats', 'token_id', 'token',
                    'analysis_id', 'morpheme_id']
    sent_id = df.sent_id.unique().item()
    grouped_rows = []
    for (token_id, analysis_id), analysis_df in df.groupby([df.token_id, df.analysis_id]):
        token = analysis_df.token.unique().item()
        morphemes = {m.morpheme_id: m for m in analysis_df.itertuples()}
        from_ids = [morphemes[morpheme_id].from_node_id for morpheme_id in sorted(morphemes)]
        to_ids = [morphemes[morpheme_id].to_node_id for morpheme_id in sorted(morphemes)]
        forms = [morphemes[morpheme_id].form for morpheme_id in sorted(morphemes)]
        lemmas = [morphemes[morpheme_id].lemma for morpheme_id in sorted(morphemes)]
        tags = [morphemes[morpheme_id].tag for morpheme_id in sorted(morphemes)]
        feats = [morphemes[morpheme_id].feats for morpheme_id in sorted(morphemes)]
        row = [sent_id, from_ids[0], to_ids[-1], '-'.join(forms), '-'.join(lemmas), '-'.join(tags), '-'.join(feats),
               token_id, token, analysis_id, 0]
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows, columns=column_names)


def add_morpheme_type(dataset):
    print("Add morpheme type")
    total_tags = defaultdict(set)
    total_multi_host_tags = set()
    for partition_type in dataset:
        for df in dataset[partition_type]:
            morpheme_types, tags, multi_host_tags = identify_morpheme_host_and_affixes(df)
            total_tags.update(tags)
            total_multi_host_tags.update(multi_host_tags)
            df['morpheme_type'] = morpheme_types
    print(total_tags)
    print(total_multi_host_tags)


def get_grouped_morpheme_type_dataset(dataset):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(get_group_morpheme_type(df))
    return grouped_dataset


def get_grouped_analysis_dataset(dataset):
    grouped_dataset = defaultdict(list)
    for partition_type in dataset:
        for df in dataset[partition_type]:
            grouped_dataset[partition_type].append(get_group_analysis(df))
    return grouped_dataset


host_tags = ['COP', 'NN', 'RB', 'BN', 'IN', 'VB', 'NNT', 'NNP', 'PRP', 'QW', 'JJ', 'CD', 'POS', 'P', 'CC',  'BNT',
             'AT', 'JJT', 'DTT', 'CDT', 'EX', 'DT', 'MD', 'INTJ', 'NEG', 'NCD']
pref_tags = ['CONJ', 'DEF', 'PREPOSITION', 'REL', 'ADVERB', 'TEMP']
suff_tags = ['S_PRN', 'S_ANP']
lattice_edge_keys = ['from_id', 'to_id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'token_id']


def save_treebank_dataset(tb_path, root_path, partition):
    tb_dataset = load_treebank(tb_path, partition)
    lattices, gold_lattices = get_dataset(tb_dataset)
    save_lattices_dataset(root_path / 'morpheme', gold_lattices, 'gold-lattices')


def save_morpheme_types(root_path, partition):
    gold_lattices = load_lattices_dataset(root_path / 'morpheme', partition, 'gold-lattices')
    # gold_lattices_with_types = add_morpheme_type(gold_lattices)
    add_morpheme_type(gold_lattices)
    save_lattices_dataset(root_path / 'morpheme-type', gold_lattices, 'gold-lattices')


def save_morpheme_type_multi_tag(root_path, partition):
    gold_lattices = load_lattices_dataset(root_path / 'morpheme-type', partition, 'gold-lattices')
    grouped_gold_lattices = get_grouped_morpheme_type_dataset(gold_lattices)
    save_lattices_dataset(root_path / 'morpheme-type', grouped_gold_lattices, 'gold-lattices-multi')


def save_token_super_tag(root_path, partition):
    gold_lattices = load_lattices_dataset(root_path / 'morpheme', partition, 'gold-lattices')
    grouped_gold_lattices = get_grouped_analysis_dataset(gold_lattices)
    save_lattices_dataset(root_path / 'token', grouped_gold_lattices, 'gold-lattices-super')


def main():
    partition = ['dev', 'test', 'train']
    root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
    tb_path = Path.home() / 'dev/onlplab/HebrewResources/HebrewTreebank/hebtb'
    save_treebank_dataset(tb_path, root_path, partition)
    save_morpheme_types(root_path, partition)
    save_morpheme_type_multi_tag(root_path, partition)
    save_token_super_tag(root_path, partition)


if __name__ == '__main__':
    main()
