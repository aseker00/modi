from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange
from utils import *
import dataset as ds
from tag_models import *
import os

from pathlib import Path
root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'

scheme = 'UD'
# scheme = 'SPMRL'
# la_name = 'ar'
la_name = 'he'
# la_name = 'tr'
if la_name == 'ar':
    tb_name = 'PADT'
if la_name == 'tr':
    tb_name = 'IMST'
else:
    if scheme == 'UD':
        tb_name = 'HTB'
    else:
        tb_name = 'HEBTB'

seq_type = 'token'
tb_root_dir_path = root_dir_path / 'tb' / scheme
data_dir_path = root_dir_path / 'data' / scheme / la_name / tb_name / 'seq' / f'{seq_type}'
out_dir_path = root_dir_path / 'out' / scheme / la_name / tb_name / 'seq' / f'{seq_type}'
os.makedirs(str(out_dir_path), exist_ok=True)
os.makedirs(str(data_dir_path), exist_ok=True)

dev_set_path = data_dir_path / 'dev-gold.pth'
test_set_path = data_dir_path / 'test-gold.pth'
train_set_path = data_dir_path / 'train-gold.pth'
char_ft_emb_path = data_dir_path / 'char-ft-gold-emb.pth'
token_ft_emb_path = data_dir_path / 'token-ft-gold-emb.pth'

if all([path.exists() for path in [dev_set_path, test_set_path, train_set_path]]):
    dev_set = torch.load(str(dev_set_path))
    test_set = torch.load(str(test_set_path))
    train_set = torch.load(str(train_set_path))
    data_vocab = ds.load_vocab(tb_root_dir_path, 'gold', la_name, tb_name)
else:
    partition = ['dev', 'test', 'train']
    token_samples, morph_samples, data_vocab = ds.load_data_samples(tb_root_dir_path, partition, 'gold', la_name, tb_name)
    token_lengths = {t: torch.tensor(token_samples[t][1], dtype=torch.long) for t in token_samples}
    token_samples = {t: torch.tensor(token_samples[t][0], dtype=torch.long) for t in token_samples}
    morph_samples = {t: torch.tensor(morph_samples[t], dtype=torch.long) for t in morph_samples}
    dev_set = TensorDataset(*[s['dev'] for s in [token_samples, token_lengths, morph_samples]])
    test_set = TensorDataset(*[s['test'] for s in [token_samples, token_lengths, morph_samples]])
    train_set = TensorDataset(*[s['train'] for s in [token_samples, token_lengths, morph_samples]])
    torch.save(dev_set, str(dev_set_path))
    torch.save(test_set, str(test_set_path))
    torch.save(train_set, str(train_set_path))

if all([path.exists() for path in [char_ft_emb_path, token_ft_emb_path]]):
    char_ft_emb = torch.load(char_ft_emb_path)
    token_ft_emb = torch.load(token_ft_emb_path)
else:
    char_ft_emb, token_ft_emb, _, _ = ds.load_ft_emb(tb_root_dir_path, ft_root_dir_path, 'gold', data_vocab, la_name, tb_name)
    torch.save(char_ft_emb, str(char_ft_emb_path))
    torch.save(token_ft_emb, str(token_ft_emb_path))

# train_set = TensorDataset(*[t[:100] for t in train_set.tensors])
train_data = DataLoader(train_set, batch_size=1, shuffle=True)
dev_data = DataLoader(dev_set, batch_size=1)
test_data = DataLoader(test_set, batch_size=1)

device = None
num_tags = len(data_vocab['tags'])
max_tag_seq_len = train_set.tensors[-1].shape[2]
token_ft_emb.weight.requires_grad = False
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=32, padding_idx=0)
seq_char_emb = TokenCharEmbedding(token_ft_emb, 0.0, char_ft_emb, 32)
seq_encoder = nn.LSTM(input_size=seq_char_emb.embedding_dim, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.0)
tag_decoder = SequenceStepDecoder(seq_char_emb.embedding_dim + tag_emb.embedding_dim, seq_encoder.hidden_size * 2, 1, 0.0, num_tags)
sos = torch.tensor([data_vocab['tag2id']['<SOS>']], dtype=torch.long, device=device)
eot = torch.tensor([data_vocab['tag2id']['<EOT>']], dtype=torch.long, device=device)
s2s = Seq2SeqClassifier(seq_char_emb, seq_encoder, tag_emb, tag_decoder, max_tag_seq_len, sos, eot)
if device is not None:
    s2s.to(device)
print(s2s)


def to_token_lattice(tag_ids, token_mask):
    if scheme == 'UD':
        return ds.tag_ids_to_ud_lattice(tag_ids, token_mask, data_vocab)
    return ds.tag_ids_to_spmrl_lattice(tag_ids, token_mask, data_vocab)


def run_data(epoch, phase, data, print_every, model, optimizer=None):
    total_loss, print_loss = 0, 0
    total_samples, print_samples = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_token_ids = batch[0]
        b_token_lengths = batch[1]
        b_morpheme_ids = batch[2]
        b_gold_tag_ids = b_morpheme_ids[:, :, :, 2]
        b_token_mask = b_token_ids[:, :, 0, 0] != 0
        b_tags_mask = b_gold_tag_ids != 0
        b_scores = model(b_token_ids, b_token_lengths, b_gold_tag_ids)
        b_loss = model.loss(b_scores, b_gold_tag_ids, b_tags_mask)
        print_loss += b_loss
        total_loss += b_loss
        b_pred_tag_ids = model.decode(b_scores)
        b_token_ids = b_token_ids.detach().cpu().numpy()
        b_token_mask = b_token_mask.detach().cpu().numpy()
        b_gold_tag_ids = b_gold_tag_ids.detach().cpu().numpy()
        b_pred_tag_ids = b_pred_tag_ids.detach().cpu().numpy()
        gold_tokens = ds.token_ids_to_tokens(b_token_ids, b_token_mask, data_vocab)
        gold_token_lattice = to_token_lattice(b_gold_tag_ids, b_token_mask)
        pred_token_lattice = to_token_lattice(b_pred_tag_ids, b_token_mask)
        print_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        total_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        if optimizer is not None:
            optimizer.step([b_loss])
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
    return total_samples


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
parameters = list(filter(lambda p: p.requires_grad, s2s.parameters()))
adam = AdamW(parameters, lr=lr)
adam = ModelOptimizer(1, adam, parameters, 5.0)
epochs = 9
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    s2s.train()
    run_data(epoch, 'train', train_data, 320, s2s, adam)
    s2s.eval()
    with torch.no_grad():
        samples = run_data(epoch, 'dev', dev_data, 32, s2s)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-dev-gold.conllu')
        test_samples = run_data(epoch, 'test', test_data, 32, s2s)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-test-gold.conllu')
