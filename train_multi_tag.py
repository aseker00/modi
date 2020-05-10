from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import numpy as np
from tqdm import trange
from utils import *
import dataset as ds
from tag_models import *
from pathlib import Path
import os


# scheme = 'UD'
scheme = 'SPMRL'
la_name = 'he'
# la_name = 'tr'
if la_name == 'he':
    if scheme == 'UD':
        tb_name = 'HTB'
    else:
        tb_name = 'HEBTB'
else:
    tb_name = 'IMST'
# multi_tag_level = 'token'
multi_tag_level = 'morpheme-type'

root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
tb_root_dir_path = root_dir_path / 'tb' / scheme
data_dir_path = root_dir_path / 'data' / scheme / la_name / tb_name / 'seq' / f'{multi_tag_level}-multi-tag'

dev_set_path = data_dir_path / 'dev-inf.pth'
test_set_path = data_dir_path / 'test-inf.pth'
train_set_path = data_dir_path / 'train-inf.pth'
char_ft_emb_path = data_dir_path / 'char-ft-emb.pth'
token_ft_emb_path = data_dir_path / 'token-ft-emb.pth'
form_ft_emb_path = data_dir_path / 'form-ft-emb.pth'
lemma_ft_emb_path = data_dir_path / 'lemma-ft-emb.pth'


if all([path.exists() for path in [dev_set_path, test_set_path, train_set_path]]):
    dev_set = torch.load(str(dev_set_path))
    test_set = torch.load(str(test_set_path))
    train_set = torch.load(str(train_set_path))
    data_vocab = ds.load_gold_multi_vocab(tb_root_dir_path, la_name, tb_name, multi_tag_level)
else:
    os.makedirs(str(data_dir_path), exist_ok=True)
    partition = ['dev', 'test', 'train']
    token_samples, morph_samples, data_vocab = ds.load_gold_multi_data_samples(tb_root_dir_path, partition, la_name, tb_name, multi_tag_level)
    token_lengths = {t: torch.tensor(token_samples[t][1], dtype=torch.long) for t in token_samples}
    token_samples = {t: torch.tensor(token_samples[t][0], dtype=torch.long) for t in token_samples}
    morph_samples = {t: torch.tensor(morph_samples[t], dtype=torch.long) for t in morph_samples}
    dev_set = TensorDataset(*[s['dev'] for s in [token_samples, token_lengths, morph_samples]])
    test_set = TensorDataset(*[s['test'] for s in [token_samples, token_lengths, morph_samples]])
    train_set = TensorDataset(*[s['train'] for s in [token_samples, token_lengths, morph_samples]])
    torch.save(dev_set, str(dev_set_path))
    torch.save(test_set, str(test_set_path))
    torch.save(train_set, str(train_set_path))
data_vocab_tags = set([tag for multi_tag in data_vocab['tags'] for tag in multi_tag.split('-')])
data_vocab_tags = [tag for tag in data_vocab_tags if tag not in data_vocab['tags']]
for tag in data_vocab_tags:
    data_vocab['tag2id'][tag] = len(data_vocab['tags'])
    data_vocab['tags'].append(tag)

if all([path.exists() for path in [char_ft_emb_path, token_ft_emb_path]]):
    char_ft_emb = torch.load(char_ft_emb_path)
    token_ft_emb = torch.load(token_ft_emb_path)
else:
    os.makedirs(str(data_dir_path), exist_ok=True)
    char_ft_emb, token_ft_emb = ds.load_gold_multi_ft_emb(tb_root_dir_path, ft_root_dir_path, data_vocab, la_name, tb_name, multi_tag_level)
    torch.save(char_ft_emb, str(char_ft_emb_path))
    torch.save(token_ft_emb, str(token_ft_emb_path))

train_data = DataLoader(train_set, batch_size=1, shuffle=True)
dev_data = DataLoader(dev_set, batch_size=1)
test_data = DataLoader(test_set, batch_size=1)


device = None
num_tags = len(data_vocab['tags'])
max_tag_seq_len = train_set.tensors[-1].shape[2]
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=100, padding_idx=0)
token_char_emb = TokenCharEmbedding(token_ft_emb, char_ft_emb, 50)
token_encoder = BatchEncoder(token_char_emb.embedding_dim, 300, 1, 0.0)
tagger = FixedSequenceClassifier(token_char_emb, token_encoder, 0.0, max_tag_seq_len, num_tags)
if device is not None:
    tagger.to(device)
print(tagger)


def to_token_lattice(tag_ids, token_mask):
    if scheme == 'UD':
        return ds.tag_ids_to_ud_lattice(tag_ids, token_mask, data_vocab)
    return ds.tag_ids_to_spmrl_lattice(tag_ids, token_mask, data_vocab)


def to_tokens(token_ids, token_mask):
    return ds.token_ids_to_tokens(token_ids, token_mask, data_vocab)


def get_num_token_tags(multi_tag_ids):
    return ds.get_num_token_tags(multi_tag_ids, data_vocab)


def to_tags(tag_ids):
    return ds.tag_ids_to_tags(tag_ids, data_vocab)


def to_tag_ids(multi_tag_ids, num_token_tags):
    multi_tag_ids_mask_idx = (multi_tag_ids != data_vocab['tag2id']['_']).nonzero()
    token_indices, tag_counts = np.unique(multi_tag_ids_mask_idx[1], axis=0, return_counts=True)
    multi_tags = np.zeros_like(multi_tag_ids)
    mask_idx = 0
    for token_idx, num_tags in zip(token_indices, tag_counts):
        for tag_idx in range(num_tags):
            tag_id = multi_tag_ids[0, token_idx, multi_tag_ids_mask_idx[2][mask_idx]]
            multi_tags[0, token_idx, tag_idx] = tag_id
            mask_idx += 1
    multi_tags = to_tags(multi_tags)
    tags = np.full_like(multi_tags, '<PAD>', shape=(multi_tags.shape[0], multi_tags.shape[1], num_token_tags))
    for batch_idx in range(multi_tags.shape[0]):
        for token_idx in range(multi_tags.shape[1]):
            tag_idx = 0
            for multi_tag_idx in range(multi_tags.shape[2]):
                multi_tag = multi_tags[batch_idx, token_idx, multi_tag_idx]
                for tag in multi_tag.split('-'):
                    tags[batch_idx, token_idx, tag_idx] = tag
                    tag_idx += 1
    return ds.tags_to_tag_ids(tags, data_vocab)


def run_data(epoch, phase, data, print_every, model, optimizer=None):
    total_loss, print_loss = 0, 0
    total_samples, print_samples = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_token_ids = batch[0]
        b_token_lengths = batch[1]
        b_morpheme_ids = batch[2]
        b_gold_multi_tag_ids = b_morpheme_ids[:, :, :, 2]
        b_token_mask = b_token_ids[:, :, 0, 0] != 0
        b_scores = model(b_token_ids, b_token_lengths)
        b_losses = model.loss(b_scores, b_gold_multi_tag_ids, b_token_mask)
        print_loss += sum(b_losses)
        total_loss += sum(b_losses)
        b_pred_multi_tag_ids = model.decode(b_scores)
        b_token_ids = b_token_ids.cpu().clone().detach().numpy()
        b_token_mask = b_token_mask.cpu().clone().detach().numpy()
        b_gold_multi_tag_ids = b_gold_multi_tag_ids.cpu().clone().detach().numpy()
        b_pred_multi_tag_ids = b_pred_multi_tag_ids.cpu().clone().detach().numpy()
        gold_tokens = to_tokens(b_token_ids, b_token_mask)
        max_token_tags_num = get_num_token_tags(b_gold_multi_tag_ids)
        max_token_tags_num = max(max_token_tags_num, get_num_token_tags(b_pred_multi_tag_ids))
        b_gold_tag_ids = to_tag_ids(b_gold_multi_tag_ids, max_token_tags_num)
        b_pred_tag_ids = to_tag_ids(b_pred_multi_tag_ids, max_token_tags_num)
        gold_token_lattice = to_token_lattice(b_gold_tag_ids, b_token_mask)
        pred_token_lattice = to_token_lattice(b_pred_tag_ids, b_token_mask)
        print_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        total_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        if optimizer is not None:
            optimizer.step(b_losses)
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch}, {phase} step {i + 1}, loss: {print_loss / print_every}')
            print_tag_metrics(print_samples, ['<PAD>'])
            print_sample_tags(print_samples[-1])
            print(ds.eval_samples(print_samples))
            print_loss = 0
            print_samples = []
    if optimizer is not None:
        optimizer.force_step()
    print(f'epoch {epoch}, {phase} total loss: {total_loss / len(data)}')
    print_tag_metrics(total_samples, ['<PAD>'])
    print(ds.eval_samples(total_samples))


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
adam = AdamW(tagger.parameters(), lr=lr)
adam = ModelOptimizer(1, adam, list(tagger.parameters()), 0.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger.train()
    run_data(epoch, 'train', train_data, 320, tagger, adam)
    tagger.eval()
    with torch.no_grad():
        run_data(epoch, 'dev', dev_data, 32, tagger)
        run_data(epoch, 'test', test_data, 32, tagger)
