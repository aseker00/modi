from models import *
import dataset as ds
import treebank as tb
from pos_tagging_utils import *
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
# import sys

root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
# sys.path.insert(0, str(root_dir_path))

partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
token_vocab, token_dataset = ds.load_token_dataset(root_dir_path, partition)
char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_token_ft_emb(root_dir_path, ft_root_dir_path, token_vocab)

train_dataset = token_dataset['train']
dev_dataset = token_dataset['dev']
test_dataset = token_dataset['test']
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)


device = None
num_tags = len(token_vocab['tags'])
tag2id = {v: i for i, v in enumerate(token_vocab['tags'])}
token_char_emb = TokenCharRNNEmbedding(char_ft_emb, 300, 1, 0.0)
token_emb = TokenEmbedding(token_ft_emb, token_char_emb, 0.7)
token_encoder = BatchTokenRNN(token_emb.embedding_dim, 300, 1, 0.0)
token_classifier = TokenClassifier(token_emb, token_encoder, 0.0, num_tags)
model = TokenMorphSeqClassifier(token_classifier)
if device is not None:
    model.to(device)


def score_tokens(tokens, chars, char_lengths, token_lengths, morphemes, tagger):
    batch_size = tokens.shape[0]
    max_len = token_lengths.max()
    tokens = tokens[:, :max_len].contiguous()
    token_indices = torch.arange(max_len, device=device).repeat(batch_size).view(batch_size, -1)
    token_masks = token_indices < token_lengths.unsqueeze(1)
    max_char_len = char_lengths.max()
    chars = chars[:, :max_len, :max_char_len].contiguous()
    char_lengths = char_lengths[:, :max_len].contiguous()
    tags = morphemes[:, :, 6:9]
    tags = tags[:, :max_len].contiguous()
    inputs = (tokens, chars, char_lengths)
    scores = tagger(inputs, token_lengths)
    losses = tagger.token_classifier.loss(scores, tags, token_masks)
    return scores, tags, tokens, losses


def score_tags(token_scores, gold_token_tags, tagger):
    batch_size = gold_token_tags.shape[0]

    # reshape [bs, seq_len, 3] tensor (last dimension represents pref,host,suff) into [bs, 3*seq_len]
    gold_token_tags = gold_token_tags.view(batch_size, -1)
    # Filter out _ tags
    gold_token_masks = gold_token_tags != tag2id['_']
    gold_tags, gold_indices = batch_mask_select(gold_token_tags, gold_token_masks)
    gold_masks = gold_tags != 0

    token_tags = tagger.token_classifier.decode(token_scores)
    token_tags = token_tags.reshape((batch_size, -1))
    token_masks = token_tags != tag2id['_']
    # stack pref, host, suff score tensors into [bs, seq_len, 3, tags] tensor
    # reshape [bs, seq_len, 3, tags] into [bs, 3*seq_len, tags]
    token_scores = torch.stack(token_scores, dim=2).reshape((batch_size, -1, token_scores[0].shape[-1]))
    tag_scores, tag_indices = batch_mask_select(token_scores, token_masks)
    tags, _ = batch_mask_select(token_tags, token_masks)
    # tags = [tags[mask] for tags, mask in zip(token_tags, token_masks)]
    # tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True)
    tag_masks = tags != tag2id['<PAD>']
    # ValueError: mask of the first timestep must all be on
    tag_masks[:, 0] = True
    tag_len = tag_scores.shape[1]
    gold_len = gold_tags.shape[1]
    fill_len = gold_len - tag_len
    if fill_len > 0:
        tag_scores = F.pad(tag_scores, [0, 0, 0, fill_len])
        tag_masks = F.pad(tag_masks, [0, fill_len])
        tag_indices = F.pad(tag_indices, [0, fill_len])
    elif fill_len < 0:
        gold_tags = F.pad(gold_tags, [0, -fill_len])
        gold_masks = F.pad(gold_masks, [0, -fill_len])
        gold_indices = F.pad(gold_indices, [0, -fill_len])
    loss = tagger.loss_crf(tag_scores, gold_tags, gold_masks)
    return tag_scores, tag_indices, tag_masks, gold_tags, gold_indices, gold_masks, loss


def run_batch(batch, tagger, optimizer):
    batch = tuple(t.to(device) for t in batch)
    tokens = batch[0]
    token_lengths = batch[1]
    chars = batch[2]
    char_lengths = batch[3]
    morphemes = batch[4]
    # tags = batch[4][:, :, 6:9]

    token_scores, gold_tags, tokens, token_losses = score_tokens(tokens, chars, char_lengths, token_lengths, morphemes, tagger)
    tag_scores, tag_indices, tag_masks, gold_tags, gold_tag_indices, gold_masks, tag_loss = score_tags(token_scores, gold_tags, tagger)
    if optimizer:
        pref_loss, host_loss, suff_loss = token_losses
        pref_loss.backward(retain_graph=True)
        host_loss.backward(retain_graph=True)
        suff_loss.backward(retain_graph=True)
        tag_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return tag_scores, tag_masks, tag_indices, tag_loss, gold_tags, gold_tag_indices, gold_masks, tokens, token_lengths


def run_epoch(epoch, phase, print_every, data, tagger, optimizer=None, max_steps=None):
    print_loss, epoch_loss = 0, 0
    print_samples, epoch_samples = [], []
    for i, batch in enumerate(data):
        step = i + 1
        tag_scores, tag_masks, tag_indices, tag_loss, gold_tags, gold_indices, gold_masks, tokens, token_lengths = run_batch(batch, tagger, optimizer)
        with torch.no_grad():
            decoded_tags = tagger.decode_crf(tag_scores, tag_masks)
        gold_token_tags = batch_re_mask_select(gold_tags, gold_indices, gold_masks, tag2id['<PAD>'], tag2id['_'], 0)
        decoded_token_tags = batch_re_mask_select(decoded_tags, tag_indices, tag_masks, tag2id['<PAD>'], tag2id['_'], 0)
        decoded_token_tags_mask = decoded_token_tags != 0
        gold_token_tags_mask = gold_token_tags != 0
        samples = to_samples(decoded_token_tags, gold_token_tags, decoded_token_tags_mask, gold_token_tags_mask, tokens, token_lengths, token_vocab)
        print_samples.append(samples)
        epoch_samples.append(samples)
        print_loss += tag_loss
        epoch_loss += tag_loss
        if step % print_every == 0:
            print(f'{phase} epoch {epoch} step {step} loss: {print_loss / print_every}')
            print_sample(print_samples[-1][0][-1], print_samples[-1][1][-1], ['<PAD>', '_'])
            print_loss = 0
            print_samples = []
        if max_steps and step == max_steps:
            break
    print(f'{phase} epoch {epoch} total loss: {epoch_loss / len(data)}')
    print_scores(epoch_samples, ['<PAD>', '_'])


for lr in [1e-3]:
    adam = AdamW(model.parameters(), lr=lr)
    for epoch in range(15):
        model.train()
        run_epoch(epoch, 'train', 32, train_dataloader, model, adam)
        with torch.no_grad():
            model.eval()
            run_epoch(epoch, 'test', 32, test_dataloader, model)
