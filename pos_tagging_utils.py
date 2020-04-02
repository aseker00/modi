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


def print_sample(pred, gold, ignore_lables):
    print([p for p in pred[0]])
    print([p for p in pred[1] if p not in ignore_lables])
    print([g for g in gold[1] if g not in ignore_lables])


def batch_mask_select(x, mask):
    y = torch.nn.utils.rnn.pad_sequence([a[m] for a, m in zip(x, mask)], batch_first=True)
    z = torch.nn.utils.rnn.pad_sequence([m.nonzero().squeeze(dim=1) for m in mask], batch_first=True)
    return y, z


def batch_re_mask_select(x, y, z, pad_id, re_id, add_extra):
    rx = []
    for sample, indices, mask in zip(x, y, z):
        r = [pad_id] * indices.shape[0]
        indices_mask = mask.nonzero().squeeze(dim=1)
        indices_len = indices[indices_mask[-1]].item()
        r[:(indices_len + 1 + add_extra)] = [re_id] * (indices_len + 1 + add_extra)
        for v, idx in zip(sample, indices_mask):
            r[indices[idx]] = v.item()
        rx.append(torch.tensor(r, dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(rx, batch_first=True)


def batch_expand_tokens(x, token_size, et_id, ex_id):
    ex = []
    for sample in x:
        e = []
        for v in sample:
            if v.item() == 0:
                continue
            e.append(v.item())
            if v.item() == et_id:
                fill_len = (len(e) // token_size + 1) * token_size - len(e)
                for i in range(fill_len):
                    e.append(ex_id)
            elif len(e) % token_size == 0:
                e.append(et_id)
        ex.append(torch.tensor(e, dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(ex, batch_first=True)


# def align_tags(tags, tag_idx, tag2id):
#     aligned_tags = []
#     for i in range(tags.shape[0]):
#         seq = []
#         # max_tag_idx = tag_idx[i][tag_mask[i]][-1]
#         max_tag_idx = tag_idx[i].shape[0]
#         cur_tag_idx = 0
#         for j in range(max_tag_idx):
#             idx = tag_idx[i][cur_tag_idx].item()
#             if j == idx:
#                 if cur_tag_idx < len(tags[i]):
#                     seq.append(tags[i][cur_tag_idx].item())
#                 cur_tag_idx += 1
#             else:
#                 seq.append(tag2id['<EOT>'])
#         aligned_tags.append(torch.tensor(seq, dtype=torch.long))
#     return aligned_tags


# def batch_mask_select(x, mask):
#     y = torch.nn.utils.rnn.pad_sequence([a[m] for a, m in zip(x, mask)], batch_first=True)
#     z = torch.nn.utils.rnn.pad_sequence([m.nonzero().squeeze(dim=1) for m in mask], batch_first=True)
#     return y, z


# def batch_expand_tokens(x, token_size, et_id, ex_id):
#     ex = []
#     for sample in x:
#         e = []
#         for v in sample:
#             if v.item() == 0:
#                 continue
#             e.append(v.item())
#             if v.item() == et_id:
#                 fill_len = (len(e) // token_size + 1) * token_size - len(e)
#                 for i in range(fill_len):
#                     e.append(ex_id)
#             elif len(e) % token_size == 0:
#                 e.append(et_id)
#         ex.append(torch.tensor(e, dtype=torch.long))
#     return torch.nn.utils.rnn.pad_sequence(ex, batch_first=True)


# def align_tags(tags, tag_idx, tag2id):
#     aligned_tags = []
#     for i in range(tags.shape[0]):
#         seq = []
#         # max_tag_idx = tag_idx[i][tag_mask[i]][-1]
#         max_tag_idx = tag_idx[i].shape[0]
#         cur_tag_idx = 0
#         for j in range(max_tag_idx):
#             idx = tag_idx[i][cur_tag_idx].item()
#             if j == idx:
#                 if cur_tag_idx < len(tags[i]):
#                     seq.append(tags[i][cur_tag_idx].item())
#                 cur_tag_idx += 1
#             else:
#                 seq.append(tag2id['<EOT>'])
#         aligned_tags.append(torch.tensor(seq, dtype=torch.long))
#     return aligned_tags


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
