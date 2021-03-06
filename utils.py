from collections import defaultdict

from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
import pandas as pd


def batch_narrow(tokens, chars, char_lengths, token_lengths, labels):
    max_token_len = token_lengths.max().item()
    max_char_len = char_lengths.max().item()
    tokens = tokens[:, :max_token_len].contiguous()
    chars = chars[:, :max_token_len, :max_char_len].contiguous()
    char_lengths = char_lengths[:, :max_token_len].contiguous()
    labels = labels[:, :max_token_len].contiguous()
    return tokens, chars, char_lengths, token_lengths, labels


def batch_expand_token_labels(samples, token_indices, max_labels_per_token, mask_label_id):
    res = []
    for sample, token_index in zip(samples, token_indices):
        seq = []
        for i, idx in enumerate(token_index):
            if sample[i] == 0:
                continue
            seq.append(sample[i].item())
            if idx:
                fill_len = max_labels_per_token - len(seq) % max_labels_per_token
                for j in range(fill_len):
                    seq.append(mask_label_id)
        res.append(torch.tensor(seq, dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(res, batch_first=True)


def batch_seq_filter_mask(scores, pred_labels, gold_labels, mask_id):
    gold_mask = gold_labels != mask_id
    gold_selected, gold_indices = batch_mask_select(gold_labels, gold_mask)
    gold_pad_mask = gold_selected != 0

    pred_mask = pred_labels != mask_id
    pred_selected, pred_indices = batch_mask_select(pred_labels, pred_mask)
    scores_selected, _ = batch_mask_select(scores, pred_mask)
    pred_pad_mask = pred_selected != 0

    # Align decoded scores and gold tags/mask
    pred_len = pred_selected.shape[1]
    gold_len = gold_selected.shape[1]
    fill_len = gold_len - pred_len
    if fill_len > 0:
        scores_selected = F.pad(scores_selected, [0, 0, 0, fill_len])
        pred_selected = F.pad(pred_selected, [0, fill_len])
        pred_pad_mask = F.pad(pred_pad_mask, [0, fill_len])
        pred_indices = F.pad(pred_indices, [0, fill_len])
    elif fill_len < 0:
        gold_selected = F.pad(gold_selected, [0, -fill_len])
        gold_pad_mask = F.pad(gold_pad_mask, [0, -fill_len])
        gold_indices = F.pad(gold_indices, [0, -fill_len])
    # ValueError: mask of the first timestep must all be on
    pred_pad_mask[:, 0] = True
    return scores_selected, pred_selected, pred_indices, pred_pad_mask, gold_selected, gold_indices, gold_pad_mask


def batch_mask_select(x, mask):
    y = torch.nn.utils.rnn.pad_sequence([a[m] for a, m in zip(x, mask)], batch_first=True)
    z = torch.nn.utils.rnn.pad_sequence([m.nonzero().squeeze(dim=1) for m in mask], batch_first=True)
    return y, z


def batch_mask_select_reconstruct(samples, indices, masks, mask_id):
    res = []
    for sample, index, mask in zip(samples, indices, masks):
        selected_sample = [0] * index.shape[0]
        mask_indices = mask.nonzero().squeeze(dim=1)
        max_len = index[mask_indices[-1]].item()
        selected_sample[:(max_len + 2)] = [mask_id] * (max_len + 2)
        for v, idx in zip(sample, mask_indices):
            selected_sample[index[idx]] = v.item()
        res.append(torch.tensor(selected_sample, dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(res, batch_first=True)


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


class ModelOptimizer:
    def __init__(self, step_every, optimizer, parameters, max_grad_norm):
        self.optimizer = optimizer
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm
        self.step_every = step_every
        self.steps = 0

    def step(self, losses):
        self.steps += 1
        # loss = loss/self.step_every
        for loss in losses[:-1]:
            loss.backward(retain_graph=True)
        loss = losses[-1]
        loss.backward()
        if self.steps % self.step_every == 0:
            self._step()

    def force_step(self):
        if self.steps % self.step_every > 0:
            self._step()

    def _step(self):
        if self.max_grad_norm > 0:
            total_norm = 0
            for p in self.parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            if total_norm > self.max_grad_norm:
                print(total_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()


def print_sample_tags(sample):
    print(f'tokens: {sample[0]}')
    print(f'gold: {sample[1][:, 2]}')
    print(f'pred: {sample[2][:, 2]}')


def print_tag_metrics(samples, remove_labels):
    gold_labels = [tag for sample in samples for tags in sample[1][:, 2, :] for tag in tags]
    pred_labels = [tag for sample in samples for tags in sample[2][:, 2, :] for tag in tags]
    labels = set(pred_labels + gold_labels)
    for label in remove_labels:
        labels.discard(label)
    print(classification_report(gold_labels, pred_labels, labels=list(labels)))
    # print(confusion_matrix(gold_tags, pred_tags))
    # precision, recall, fscore, support = precision_recall_fscore_support(gold_tags, pred_tags)


def validate_baseline_tokens():
    g = pd.read_csv('UD/he/HTB/train-gold.lattices.csv', index_col=0, keep_default_na=False)
    u = pd.read_csv('UD/he/HTB/train-udpipe.lattices.csv', index_col=0, keep_default_na=False)

    gd = defaultdict(dict)
    for i, x in g.groupby(g.sent_id):
        for j, y in x.groupby(x.token_id):
            gd[i][j] = str(y.token.unique())

    ud = defaultdict(dict)
    for i, x in u.groupby(u.sent_id):
        for j, y in x.groupby(x.token_id):
            ud[i][j] = str(y.token.unique())

    for a, b in zip(gd, ud):
        if gd[a] != ud[b]:
            print(a, b)
