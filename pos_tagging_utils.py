from sklearn.metrics import classification_report
import torch.nn.functional as F
import torch


def print_scores(samples, ignore_lables):
    pred_tags = [tag for batch in samples for sample in batch[0] for tag in sample[1]]
    gold_tags = [tag for batch in samples for sample in batch[1] for tag in sample[1]]
    # precision, recall, fscore, support = precision_recall_fscore_support(gold_tags, pred_tags)
    labels = set(pred_tags + gold_tags)
    for label in ignore_lables:
        labels.discard(label)
    print(classification_report(gold_tags, pred_tags, labels=list(labels)))
    # print(confusion_matrix(gold_tags, pred_tags))


def to_samples(pred_tag_seq, gold_tag_seq, pred_tag_seq_mask, gold_tag_seq_mask, token_seq, token_lengths, vocab):
    token_ids = [seq[:len] for seq, len in zip(token_seq, token_lengths)]
    token_samples = [[vocab['tokens'][i] for i in x] for x in token_ids]
    max_tag_len = max(pred_tag_seq.shape[1], gold_tag_seq.shape[1])
    pred_tag_ids = [F.pad(seq[mask], [0, max_tag_len - seq[mask].shape[0]]) for seq, mask in zip(pred_tag_seq, pred_tag_seq_mask)]
    gold_tag_ids = [F.pad(seq[mask], [0, max_tag_len - seq[mask].shape[0]]) for seq, mask in zip(gold_tag_seq, gold_tag_seq_mask)]
    pred_tag_samples = [[vocab['tags'][i] for i in x] for x in pred_tag_ids]
    gold_tag_samples = [[vocab['tags'][i] for i in x] for x in gold_tag_ids]
    pred_samples = [(token_sample, tag_sample) for token_sample, tag_sample in zip(token_samples, pred_tag_samples)]
    gold_samples = [(token_sample, tag_sample) for token_sample, tag_sample in zip(token_samples, gold_tag_samples)]
    return pred_samples, gold_samples


def print_sample(pred, gold):
    print([p for p in pred[0]])
    print([p for p in pred[1] if p != '<PAD>'])
    print([g for g in gold[1] if g != '<PAD>'])


def align_token_tags(tags, tag_idx, tag_mask, token_lengths, max_token_tags, tag2id):
    token_tags = []
    for j in range(token_lengths.shape[0]):
        # [num_tokens * 3] - (pref,host,suff) for each token
        seq = torch.ones(token_lengths[j] * max_token_tags, dtype=torch.long) * tag2id['_']
        # tag indices in the token sequence
        indices = tag_idx[j][tag_mask[j]]
        # tag ids
        tids = tags[j][tag_mask[j]]
        # set tag ids in the token sequence
        for idx, tid in zip(indices, tids):
            if idx < len(seq):
                seq[idx] = tid
        token_tags.append(seq)
    return token_tags


def index_token_tag_seq(tag_seq, num_token_tags, tag2id):
    idx = []
    for seq in tag_seq:
        seq_indices = []
        cur_idx = 0
        for tag in seq:
            seq_indices.append(cur_idx)
            if tag == tag2id['<EOT>']:
                cur_idx = ((cur_idx + num_token_tags) // num_token_tags) * num_token_tags
            else:
                cur_idx += 1
        idx.append(seq_indices)
    return idx
