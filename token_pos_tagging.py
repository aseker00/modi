from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader

from models import *
import dataset as ds
import treebank as tb
from pos_tagging_utils import *


from pathlib import Path
root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
token_vocab, token_dataset = ds.load_token_dataset(root_dir_path, partition)
char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_token_ft_emb(root_dir_path, ft_root_dir_path, token_vocab)

train_dataset = token_dataset['train']
dev_dataset = token_dataset['dev']
test_dataset = token_dataset['test']
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

num_tags = len(token_vocab['tags'])
token_char_emb = TokenCharRNNEmbedding(char_ft_emb, 60, 1, 0.0)
token_emb = TokenEmbedding(token_ft_emb, token_char_emb, 0.1)
token_encoder = TokenRNN(token_emb.embedding_dim, 100, 2, 0.1)
model = TokenClassifier(token_emb, token_encoder, 0.1, num_tags)
device = None


def run_batch(batch, tagger, optimizer):
    batch = tuple(t.to(device) for t in batch)
    token_seq = batch[0]
    token_lengths = batch[1]
    char_seq = batch[2]
    char_lengths = batch[3]
    tag_seq = batch[4][:, :, 6:9]

    max_token_seq = token_lengths.max()
    packed_token_seq = token_seq[:, :max_token_seq].contiguous()

    max_char_seq = char_lengths.max()
    packed_char_seq = char_seq[:, :max_token_seq, :max_char_seq].contiguous()
    packed_char_lengths = char_lengths[:, :max_token_seq].contiguous()

    batch_size = packed_token_seq.shape[0]
    packed_seq_length = packed_token_seq.shape[1]
    packed_tag_seq = tag_seq[:, :max_token_seq].contiguous()
    packed_seq_idx = torch.arange(packed_seq_length, device=device).repeat(batch_size).view(batch_size, -1)
    packed_seq_mask = packed_seq_idx < token_lengths.unsqueeze(1)
    packed_input_seq = (packed_token_seq, packed_char_seq, packed_char_lengths)

    tag_scores = tagger(packed_input_seq, token_lengths)
    tag_losses = tagger.loss(tag_scores, packed_tag_seq, packed_seq_mask)
    if optimizer:
        pref_loss, host_loss, suff_loss = tag_losses
        pref_loss.backward(retain_graph=True)
        host_loss.backward(retain_graph=True)
        suff_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return tag_scores, packed_tag_seq, tag_losses, packed_seq_mask, packed_token_seq, token_lengths


def run_epoch(epoch, phase, print_every, data, tagger, optimizer=None):
    print_loss, epoch_loss = 0, 0
    print_samples, epoch_samples = [], []
    for i, batch in enumerate(data):
        step = i + 1
        tag_scores, gold_tags, tag_losses, mask, token_seq, seq_lengths = run_batch(batch, tagger, optimizer)
        with torch.no_grad():
            pred_tag_seq = tagger.decode(tag_scores)
        gold_tag_seq = gold_tags.view(gold_tags.shape[0], -1)
        pred_tag_seq = pred_tag_seq.view(pred_tag_seq.shape[0], -1)
        mask_tag_seq = torch.stack([mask, mask, mask], dim=2).view(mask.shape[0], -1)
        # pred_tags = pred_tags.view(pred_tags.shape[0], pred_tags[1] * pred_tags[2], -1)
        # .transpose(dim0=1, dim1=2).view(pred_tags.shape[0], pred_tags[1] * pred_tags[2], -1)
        samples = to_samples(pred_tag_seq, gold_tag_seq, mask_tag_seq, mask_tag_seq, token_seq, seq_lengths, token_vocab)
        print_samples.append(samples)
        epoch_samples.append(samples)
        print_loss += sum(tag_losses)
        epoch_loss += sum(tag_losses)
        if step % print_every == 0:
            print(f'{phase} epoch {epoch} step {step} loss: {print_loss / print_every}')
            print_sample(print_samples[-1][0][-1], print_samples[-1][1][-1])
            print_loss = 0
            print_samples = []
    print(f'{phase} epoch {epoch} total loss: {epoch_loss / len(data)}')
    print_scores(epoch_samples, ['_', '<PAD>'])


for lr in [1e-2, 1e-3, 1e-3]:
    adam = AdamW(model.parameters(), lr=lr)
    for epoch in range(3):
        model.train()
        run_epoch(epoch, 'train', 10, train_dataloader, model, adam)
        with torch.no_grad():
            model.eval()
            run_epoch(epoch, 'test', 1, test_dataloader, model)
