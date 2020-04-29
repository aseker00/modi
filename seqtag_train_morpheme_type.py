from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange
import numpy as np
from utils import *
import seqtag_dataset as ds
from seqtag_models import *
from pathlib import Path


root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
ft_root_path = Path.home() / 'dev/aseker00/fasttext'
seq_type = 'morpheme-type'
dev_set_path = Path(f'{seq_type}_dev.pth')
test_set_path = Path(f'{seq_type}_test.pth')
train_set_path = Path(f'{seq_type}_train.pth')
char_ft_emb_path = Path('char_ft_emb.pth')
token_ft_emb_path = Path('token_ft_emb.pth')

if dev_set_path.exists() and test_set_path.exists() and train_set_path.exists():
    dev_set = torch.load(str(dev_set_path))
    test_set = torch.load(str(test_set_path))
    train_set = torch.load(str(train_set_path))
    vocab = ds.load_vocab(root_path / f'{seq_type}/vocab')
else:
    partition = ['dev', 'test', 'train']
    token_arr, morph_arr, vocab = ds.load_samples(root_path, partition, seq_type, 'fixed')
    token_lengths = {t: torch.tensor(token_arr[t][1], dtype=torch.long, requires_grad=False) for t in token_arr}
    token_samples = {t: torch.tensor(token_arr[t][0], dtype=torch.long) for t in token_arr}
    morph_samples = {t: torch.tensor(morph_arr[t], dtype=torch.long) for t in morph_arr}
    dev_set = TensorDataset(token_samples['dev'], token_lengths['dev'], morph_samples['dev'])
    test_set = TensorDataset(token_samples['test'], token_lengths['test'], morph_samples['test'])
    train_set = TensorDataset(token_samples['train'], token_lengths['train'], morph_samples['train'])
    torch.save(dev_set, str(dev_set_path))
    torch.save(test_set, str(test_set_path))
    torch.save(train_set, str(train_set_path))
if char_ft_emb_path.exists() and token_ft_emb_path.exists():
    char_ft_emb = torch.load(char_ft_emb_path)
    token_ft_emb = torch.load(token_ft_emb_path)
else:
    char_ft_emb, token_ft_emb, _, _ = ds.load_ft_vec(root_path / f'{seq_type}/vocab', ft_root_path, vocab)
    torch.save(char_ft_emb, str(char_ft_emb_path))
    torch.save(token_ft_emb, str(token_ft_emb_path))
train_data = DataLoader(train_set, batch_size=1, shuffle=True)
dev_data = DataLoader(dev_set, batch_size=1)
test_data = DataLoader(test_set, batch_size=1)

device = None
num_tags = len(vocab['tags'])
max_tag_seq_len = train_set.tensors[-1].shape[2]
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=100, padding_idx=0)
token_char_emb = TokenCharEmbedding(token_ft_emb, char_ft_emb, 50)
token_encoder = BatchEncoder(token_char_emb.embedding_dim, 300, 1, 0.0)
tagger = FixedSequenceClassifier(token_char_emb, token_encoder, 0.0, max_tag_seq_len, num_tags)
if device is not None:
    tagger.to(device)
print(tagger)


def to_lattice_data(tokens, token_mask, morphemes, tags):
    # token_sample = tokens[:, :, 0, 0][token_mask]
    # lattice_sample = pack_lattice(lattice, token_mask, analysis_indices)
    # return ds.lattice_to_data(token_sample.cpu().numpy(), lattice_sample.cpu().numpy(), vocab)
    pass


split_multi_tags = np.vectorize(lambda x: len(x.split('-')))


def to_tags_arr(tag_ids, token_mask, vocab):
    token_tag_ids = tag_ids[token_mask]
    token_tag_ids_mask_idx = (token_tag_ids != vocab['tag2id']['_']).nonzero()
    token_indices, tag_counts = token_tag_ids_mask_idx[:, 0].unique_consecutive(dim=0, return_counts=True)
    multi_tags = torch.zeros_like(token_tag_ids)
    mask_idx = 0
    for token_idx, num_tags in zip(token_indices, tag_counts):
        for tag_idx in range(num_tags):
            tag_id = token_tag_ids[token_idx, token_tag_ids_mask_idx[mask_idx, 1]]
            multi_tags[token_idx, tag_idx] = tag_id
            mask_idx += 1
    multi_tags = ds.to_tag_vec(multi_tags.cpu().numpy(), vocab)
    num_tags = split_multi_tags(multi_tags).sum(axis=1)
    tags = np.full((multi_tags.shape[0], np.max(num_tags)), fill_value=vocab['tags'][0], dtype=multi_tags.dtype)
    for token_idx in range(multi_tags.shape[0]):
        tag_idx = 0
        for multi_tag_idx in range(multi_tags.shape[1]):
            multi_tag = multi_tags[token_idx, multi_tag_idx]
            for tag in multi_tag.split('-'):
                tags[token_idx, tag_idx] = tag
                tag_idx += 1
    return tags


def get_fixed_samples(samples):
    fixed_seq_len = max([sample[1].shape[1] for sample in samples] + [sample[2].shape[1] for sample in samples])
    return [get_fixed_sample(sample, fixed_seq_len) for sample in samples]


def get_fixed_sample(sample, max_seq_len):
    fixed_gold_labels = np.pad(sample[1], ((0, 0), (0, max_seq_len - sample[1].shape[1])), mode='edge')
    fixed_pred_labels = np.pad(sample[2], ((0, 0), (0, max_seq_len - sample[2].shape[1])), mode='edge')
    return sample[0], fixed_gold_labels, fixed_pred_labels


def run_data(epoch, phase, data, print_every, model, optimizer=None):
    total_loss, print_loss = 0, 0
    total_samples, print_samples = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_tokens = batch[0]
        b_token_lengths = batch[1]
        b_morphemes = batch[2]
        b_gold_tags = b_morphemes[:, :, :, 2]
        b_token_mask = b_tokens[:, :, 0, 0] != 0
        # [b_max_tokens, b_max_chars] = b_token_lengths[:, :].max(dim=1)[0][0].tolist()
        b_scores = model(b_tokens, b_token_lengths)
        b_losses = model.loss(b_scores, b_gold_tags, b_token_mask)
        print_loss += sum(b_losses)
        total_loss += sum(b_losses)
        b_pred_tags = model.decode(b_scores)
        gold_tokens_arr = to_tokens_arr(b_tokens, b_token_mask, vocab)
        gold_labels_arr = to_tags_arr(b_gold_tags, b_token_mask, vocab)
        pred_labels_arr = to_tags_arr(b_pred_tags, b_token_mask, vocab)
        print_samples.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        total_samples.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        if optimizer is not None:
            optimizer.step(b_losses)
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch}, {phase} step {i + 1}, loss: {print_loss / print_every}')
            fixed_samples = get_fixed_samples(print_samples)
            print_label_metrics(fixed_samples, ['<PAD>'])
            print_sample_labels(fixed_samples[-1])
            print_loss = 0
            print_samples = []
    if optimizer is not None:
        optimizer.force_step()
    print(f'epoch {epoch}, {phase} total loss: {total_loss / len(data)}')
    fixed_samples = get_fixed_samples(total_samples)
    print_label_metrics(fixed_samples, ['<PAD>'])


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
adam = AdamW(tagger.parameters(), lr=lr)
adam = ModelOptimizer(1, adam, tagger.parameters(), 0.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger.train()
    run_data(epoch, 'train', train_data, 320, tagger, adam)
    tagger.eval()
    with torch.no_grad():
        run_data(epoch, 'dev', dev_data, 32, tagger)
        run_data(epoch, 'test', test_data, 32, tagger)
