from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange
from seqtag_utils import *
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
vocab_tags = set([tag for multi_tag in vocab['tags'] for tag in multi_tag.split('-')])
vocab_tags = [tag for tag in vocab_tags if tag not in vocab['tags']]
for tag in vocab_tags:
    vocab['tag2id'][tag] = len(vocab['tags'])
    vocab['tags'].append(tag)

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


def get_num_token_tags(multi_tag_ids):
    multi_tags = ds.to_tag_vec(multi_tag_ids, vocab)
    return ds.get_multi_tags_len(multi_tags).sum(axis=2).max()


def to_tag_ids(multi_tag_ids, num_token_tags):
    multi_tag_ids_mask_idx = (multi_tag_ids != vocab['tag2id']['_']).nonzero()
    token_indices, tag_counts = np.unique(multi_tag_ids_mask_idx[1], axis=0, return_counts=True)
    multi_tags = np.zeros_like(multi_tag_ids)
    mask_idx = 0
    for token_idx, num_tags in zip(token_indices, tag_counts):
        for tag_idx in range(num_tags):
            tag_id = multi_tag_ids[0, token_idx, multi_tag_ids_mask_idx[2][mask_idx]]
            # First tag in each token must have a value (non <XXX> tag)
            if tag_idx == 0 and tag_id == vocab['tag2id']['<PAD>']:
                tag_id = vocab['tag2id']['_']
            multi_tags[0, token_idx, tag_idx] = tag_id
            mask_idx += 1
    multi_tags = ds.to_tag_vec(multi_tags, vocab)
    tags = np.full_like(multi_tags, shape=(multi_tags.shape[0], multi_tags.shape[1], num_token_tags), fill_value='<PAD>')
    for batch_idx in range(multi_tags.shape[0]):
        for token_idx in range(multi_tags.shape[1]):
            tag_idx = 0
            for multi_tag_idx in range(multi_tags.shape[2]):
                multi_tag = multi_tags[batch_idx, token_idx, multi_tag_idx]
                for tag in multi_tag.split('-'):
                    tags[batch_idx, token_idx, tag_idx] = tag
                    tag_idx += 1
    return ds.to_tag_id_vec(tags, vocab)


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
        # [b_max_tokens, b_max_chars] = b_token_lengths[:, :].max(dim=1)[0][0].tolist()
        b_scores = model(b_token_ids, b_token_lengths)
        b_losses = model.loss(b_scores, b_gold_multi_tag_ids, b_token_mask)
        print_loss += sum(b_losses)
        total_loss += sum(b_losses)
        b_pred_multi_tag_ids = model.decode(b_scores)
        b_token_ids = b_token_ids.cpu().numpy()
        b_token_mask = b_token_mask.cpu().numpy()
        b_gold_multi_tag_ids = b_gold_multi_tag_ids.cpu().numpy()
        b_pred_multi_tag_ids = b_pred_multi_tag_ids.cpu().numpy()
        gold_tokens = to_tokens(b_token_ids, b_token_mask, vocab)
        max_token_tags_num = get_num_token_tags(b_gold_multi_tag_ids)
        max_token_tags_num = max(max_token_tags_num, get_num_token_tags(b_pred_multi_tag_ids))
        b_gold_tag_ids = to_tag_ids(b_gold_multi_tag_ids, max_token_tags_num)
        b_pred_tag_ids = to_tag_ids(b_pred_multi_tag_ids, max_token_tags_num)
        gold_token_lattice = to_token_lattice(b_gold_tag_ids, b_token_mask, vocab)
        pred_token_lattice = to_token_lattice(b_pred_tag_ids, b_token_mask, vocab)
        print_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        total_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        if optimizer is not None:
            optimizer.step(b_losses)
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch}, {phase} step {i + 1}, loss: {print_loss / print_every}')
            print_tag_metrics(print_samples, ['<PAD>'])
            print_sample_tags(print_samples[-1])
            print(eval_samples(print_samples))
            print_loss = 0
            print_samples = []
    if optimizer is not None:
        optimizer.force_step()
    print(f'epoch {epoch}, {phase} total loss: {total_loss / len(data)}')
    print_tag_metrics(total_samples, ['<PAD>'])
    print(eval_samples(total_samples))


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
adam = AdamW(tagger.parameters(), lr=lr)
adam = ModelOptimizer(1, adam, tagger.parameters(), 0.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger.train()
    run_data(epoch, 'train', train_data, 32, tagger, adam)
    tagger.eval()
    with torch.no_grad():
        run_data(epoch, 'dev', dev_data, 32, tagger)
        run_data(epoch, 'test', test_data, 32, tagger)
