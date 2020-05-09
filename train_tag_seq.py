from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange
from utils import *
import dataset as ds
from tag_models import *
from pathlib import Path
import os


la_name = 'tr'
tb_name = 'IMST'
# la_name = 'he'
# tb_name = 'HTB'
# tb_name = 'HEBTB'
scheme = 'UD'
# scheme = 'SPMRL'
root_path = Path.home() / 'dev/aseker00/modi'
tb_root_dir_path = root_path / 'tb' / scheme
data_dir_path = root_path / 'data' /scheme / la_name / tb_name

ft_root_path = Path.home() / 'dev/aseker00/fasttext'
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
    data_vocab = ds.load_gold_vocab(tb_root_dir_path, la_name, tb_name)
else:
    os.makedirs(str(data_dir_path), exist_ok=True)
    partition = ['dev', 'test', 'train']
    token_samples, morph_samples, data_vocab = ds.load_gold_data_samples(tb_root_dir_path, partition, la_name, tb_name)
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
    os.makedirs(str(data_dir_path), exist_ok=True)
    char_ft_emb, token_ft_emb = ds.load_gold_ft_emb(tb_root_dir_path, ft_root_path, data_vocab, la_name, tb_name)
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
token_encoder = nn.LSTM(input_size=token_char_emb.embedding_dim, hidden_size=300, num_layers=1, bidirectional=True,
                        batch_first=True, dropout=0.0)
tag_decoder = SequenceStepDecoder(token_char_emb.embedding_dim + tag_emb.embedding_dim, token_encoder.hidden_size * 2, 1, 0.0, num_tags)
sos = torch.tensor([data_vocab['tag2id']['<SOS>']], dtype=torch.long, device=device)
eot = torch.tensor([data_vocab['tag2id']['<EOT>']], dtype=torch.long, device=device)
s2s = Seq2SeqClassifier(token_char_emb, token_encoder, tag_emb, tag_decoder, max_tag_seq_len, sos, eot)
if device is not None:
    s2s.to(device)
print(s2s)


def to_token_lattice(tag_ids, token_mask):
    if scheme == 'UD':
        return ds.tag_ids_to_ud_lattice(tag_ids, token_mask, data_vocab)
    return ds.tag_ids_to_spmrl_lattice(tag_ids, token_mask, data_vocab)


def to_tokens(token_ids, token_mask):
    return ds.token_ids_to_tokens(token_ids, token_mask, data_vocab)


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
        b_token_ids = b_token_ids.cpu().numpy()
        b_token_mask = b_token_mask.cpu().numpy()
        b_gold_tag_ids = b_gold_tag_ids.cpu().numpy()
        b_pred_tag_ids = b_pred_tag_ids.cpu().numpy()
        gold_tokens = to_tokens(b_token_ids, b_token_mask)
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


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
adam = AdamW(s2s.parameters(), lr=lr)
adam = ModelOptimizer(1, adam, s2s.parameters(), 0.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    s2s.train()
    run_data(epoch, 'train', train_data, 320, s2s, adam)
    s2s.eval()
    with torch.no_grad():
        run_data(epoch, 'dev', dev_data, 32, s2s)
        run_data(epoch, 'test', test_data, 32, s2s)