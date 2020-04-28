from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange

from utils import *
import seqtag_dataset as ds
from seqtag_models import *
from pathlib import Path


root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
ft_root_path = Path.home() / 'dev/aseker00/fasttext'
seq_type = 'morpheme'
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
    token_arr, morph_arr, vocab = ds.load_samples(root_path, partition, seq_type, 'var')
    token_lengths = {t: torch.tensor(token_arr[t][1], dtype=torch.long) for t in token_arr}
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
token_encoder = nn.LSTM(input_size=token_char_emb.embedding_dim, hidden_size=300, num_layers=1, bidirectional=True,
                        batch_first=True, dropout=0.0)
tag_decoder = SequenceStepDecoder(token_char_emb.embedding_dim + tag_emb.embedding_dim, token_encoder.hidden_size * 2, 1, 0.0, num_tags)
sos = torch.tensor([vocab['tag2id']['<SOS>']], dtype=torch.long, device=device)
eot = torch.tensor([vocab['tag2id']['<EOT>']], dtype=torch.long, device=device)
s2s = Seq2SeqClassifier(token_char_emb, token_encoder, tag_emb, tag_decoder, max_tag_seq_len, sos, eot)
if device is not None:
    s2s.to(device)
print(s2s)


def to_lattice_data(tokens, token_mask, morphemes, tags):
    # token_sample = tokens[:, :, 0, 0][token_mask]
    # lattice_sample = pack_lattice(lattice, token_mask, analysis_indices)
    # return ds.lattice_to_data(token_sample.cpu().numpy(), lattice_sample.cpu().numpy(), vocab)
    pass


def to_tags_arr(tag_ids, token_mask):
    masked_token_tag_ids = tag_ids[token_mask]
    return ds.to_tag_vec(masked_token_tag_ids.cpu().numpy(), vocab)


def to_tokens_arr(token_ids, token_mask):
    masked_token_ids = token_ids[:, :, 0, 0][token_mask]
    return ds.to_token_vec(masked_token_ids.cpu().numpy(), vocab)


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
        b_tags_mask = b_gold_tags != 0
        # [b_max_tokens, b_max_chars] = b_token_lengths[:, :].max(dim=1)[0][0].tolist()
        b_scores = model(b_tokens, b_token_lengths, b_gold_tags)
        b_loss = model.loss(b_scores, b_gold_tags, b_tags_mask)
        print_loss += b_loss
        total_loss += b_loss
        b_pred_tags = model.decode(b_scores)
        gold_tokens_arr = to_tokens_arr(b_tokens, b_token_mask)
        gold_labels_arr = to_tags_arr(b_gold_tags, b_token_mask)
        pred_labels_arr = to_tags_arr(b_pred_tags, b_token_mask)
        print_samples.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        total_samples.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        if optimizer is not None:
            optimizer.step([b_loss])
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch}, {phase} step {i + 1}, loss: {print_loss / print_every}')
            print_label_metrics(print_samples, ['<PAD>', '<EOT>'])
            print_sample_labels(print_samples[-1])
            print_loss = 0
            print_samples = []
    if optimizer is not None:
        optimizer.force_step()
    print(f'epoch {epoch}, {phase} total loss: {total_loss / len(data)}')
    print_label_metrics(total_samples, ['<PAD>', '<EOT>'])


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
adam = AdamW(s2s.parameters(), lr=lr)
adam = ModelOptimizer(1, adam, s2s.parameters(), 0.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    s2s.train()
    run_data(epoch, 'train', train_data, 32, s2s, adam)
    s2s.eval()
    with torch.no_grad():
        run_data(epoch, 'dev', dev_data, 32, s2s)
        run_data(epoch, 'test', test_data, 32, s2s)