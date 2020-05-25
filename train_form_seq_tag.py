from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange
from utils import *
import dataset as ds
from tag_models import *
import os
import numpy as np

from pathlib import Path
root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'

scheme = 'UD'
# scheme = 'SPMRL'
# la_name = 'ar'
la_name = 'en'
# la_name = 'he'
# la_name = 'tr'
if la_name == 'ar':
    tb_name = 'PADT'
elif la_name == 'en':
    tb_name = 'EWT'
elif la_name == 'tr':
    tb_name = 'IMST'
else:
    if scheme == 'UD':
        tb_name = 'HTB'
    else:
        tb_name = 'HEBTB'

seq_type = 'form'
tb_root_dir_path = root_dir_path / 'tb' / scheme
data_dir_path = root_dir_path / 'data' / scheme / la_name / tb_name / 'seq' / f'{seq_type}'
out_dir_path = root_dir_path / 'out' / scheme / la_name / tb_name / 'seq' / f'{seq_type}'
os.makedirs(str(out_dir_path), exist_ok=True)
os.makedirs(str(data_dir_path), exist_ok=True)

gold_dev_set_path = data_dir_path / f'dev-gold.pth'
gold_test_set_path = data_dir_path / f'test-gold.pth'
gold_train_set_path = data_dir_path / f'train-gold.pth'
udpipe_dev_set_path = data_dir_path / f'dev-udpipe.pth'
udpipe_test_set_path = data_dir_path / f'test-udpipe.pth'
char_ft_emb_path = data_dir_path / f'char-ft-gold-emb.pth'
form_ft_emb_path = data_dir_path / f'form-ft-gold-emb.pth'

if all([path.exists() for path in [gold_dev_set_path, gold_test_set_path, gold_train_set_path, udpipe_dev_set_path, udpipe_test_set_path]]):
    gold_dev_set = torch.load(gold_dev_set_path)
    gold_test_set = torch.load(gold_test_set_path)
    gold_train_set = torch.load(gold_train_set_path)
    udpipe_dev_set = torch.load(udpipe_dev_set_path)
    udpipe_test_set = torch.load(udpipe_test_set_path)
    data_vocab = ds.load_vocab(tb_root_dir_path, 'udpipe', la_name, tb_name, seq_type)
else:
    partition = ['dev', 'test', 'train']
    token_samples, gold_morph_samples, udpipe_morph_samples, data_vocab = ds.load_data_samples(tb_root_dir_path, partition, 'udpipe', la_name, tb_name)
    token_lengths = {t: torch.tensor(token_samples[t][1], dtype=torch.long) for t in token_samples}
    token_samples = {t: torch.tensor(token_samples[t][0], dtype=torch.long) for t in token_samples}
    gold_morph_samples = {t: torch.tensor(gold_morph_samples[t], dtype=torch.long) for t in gold_morph_samples}
    udpipe_morph_samples = {t: torch.tensor(udpipe_morph_samples[t], dtype=torch.long) for t in udpipe_morph_samples}

    gold_dev_set = TensorDataset(*[s['dev'] for s in [token_samples, token_lengths, gold_morph_samples, gold_morph_samples]])
    gold_test_set = TensorDataset(*[s['test'] for s in [token_samples, token_lengths, gold_morph_samples, gold_morph_samples]])
    gold_train_set = TensorDataset(*[s['train'] for s in [token_samples, token_lengths, gold_morph_samples, gold_morph_samples]])
    udpipe_dev_set = TensorDataset(*[s['dev'] for s in [token_samples, token_lengths, udpipe_morph_samples, gold_morph_samples]])
    udpipe_test_set = TensorDataset(*[s['test'] for s in [token_samples, token_lengths, udpipe_morph_samples, gold_morph_samples]])

    torch.save(gold_dev_set, gold_dev_set_path)
    torch.save(gold_test_set, gold_test_set_path)
    torch.save(gold_train_set, gold_train_set_path)
    torch.save(udpipe_dev_set, udpipe_dev_set_path)
    torch.save(udpipe_test_set, udpipe_test_set_path)

if all([path.exists() for path in [char_ft_emb_path, form_ft_emb_path]]):
    char_ft_emb = torch.load(char_ft_emb_path)
    form_ft_emb = torch.load(form_ft_emb_path)
else:
    char_ft_emb, _, form_ft_emb, _ = ds.load_ft_emb(tb_root_dir_path, ft_root_dir_path, 'udpipe', data_vocab, la_name, tb_name)
    torch.save(char_ft_emb, str(char_ft_emb_path))
    torch.save(form_ft_emb, str(form_ft_emb_path))

gold_train_set = TensorDataset(*[t[:100] for t in gold_train_set.tensors])
gold_train_data = DataLoader(gold_train_set, batch_size=1, shuffle=True)
gold_dev_data = DataLoader(gold_dev_set, batch_size=1)
gold_test_data = DataLoader(gold_test_set, batch_size=1)
udpipe_dev_data = DataLoader(udpipe_dev_set, batch_size=1)
udpipe_test_data = DataLoader(udpipe_test_set, batch_size=1)

device = None
num_tags = len(data_vocab['tags'])
form_ft_emb.weight.requires_grad = False
seq_char_emb = TokenCharEmbedding(form_ft_emb, 0.0, char_ft_emb, 32)
seq_encoder = BatchEncoder(seq_char_emb.embedding_dim, 64, 2, 0.0)
tagger = FixedSequenceClassifier(seq_char_emb, seq_encoder, 0.0, 1, num_tags)
if device is not None:
    tagger.to(device)
print(tagger)


def to_token_lattice(tag_ids, token_mask):
    if scheme == 'UD':
        return ds.tag_ids_to_ud_lattice(tag_ids, token_mask, data_vocab)
    return ds.tag_ids_to_spmrl_lattice(tag_ids, token_mask, data_vocab)


def get_form_input_seq(form_ids):
    form_vec = np.vectorize(lambda x: data_vocab['forms'][x])
    form_len = np.vectorize(len)
    forms = form_vec(form_ids.detach().cpu().numpy())
    char_ids = [torch.tensor([data_vocab['char2id'][c] for c in str(t)], dtype=torch.long, device=device) for t in forms]
    char_ids = pad_sequence(char_ids, batch_first=True)
    seq_ids = torch.stack([form_ids[:, None].repeat([1, char_ids.shape[1]]), char_ids], dim=2)
    form_lengths = torch.tensor(form_len(forms), dtype=torch.long, device=device)
    seq_lengths = torch.stack([torch.zeros_like(form_lengths), form_lengths], dim=1)
    seq_lengths[0, 0] = form_ids.shape[0]
    return seq_ids.unsqueeze(0), seq_lengths.unsqueeze(0)


def run_data(epoch, phase, data, print_every, model, optimizer=None):
    total_loss, print_loss = 0, 0
    total_samples, print_samples = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)

        b_token_ids = batch[0]
        # b_token_lengths = batch[1]
        b_token_mask = b_token_ids[:, :, 0, 0] != 0

        # Evaluation gold segments
        b_eval_morpheme_ids = batch[3]
        b_eval_gold_tag_ids = b_eval_morpheme_ids[:, :, :, 2]
        b_eval_gold_form_ids = b_eval_morpheme_ids[:, :, :, 0]
        b_eval_gold_form_mask = b_eval_gold_form_ids > 2
        b_eval_gold_form_mask_ids = b_eval_gold_form_mask.nonzero()
        b_eval_form_tag_ids = b_eval_gold_tag_ids[b_eval_gold_form_mask_ids[:, 0], b_eval_gold_form_mask_ids[:, 1], b_eval_gold_form_mask_ids[:, 2]]
        b_eval_form_tag_ids = b_eval_form_tag_ids.unsqueeze(0).unsqueeze(2)
        # b_eval_form_ids = b_eval_gold_form_ids[b_eval_gold_form_mask_ids[:, 0], b_eval_gold_form_mask_ids[:, 1], b_eval_gold_form_mask_ids[:, 2]]
        # b_eval_form_token_ids, b_eval_form_token_lengths = get_form_input_seq(b_eval_form_ids)
        # b_eval_form_token_mask = b_eval_form_token_ids[:, :, 0, 0] != 0

        # Training segments
        b_morpheme_ids = batch[2]
        b_gold_tag_ids = b_morpheme_ids[:, :, :, 2]
        b_gold_form_ids = b_morpheme_ids[:, :, :, 0]
        b_gold_form_mask = b_gold_form_ids > 2
        b_gold_form_mask_ids = b_gold_form_mask.nonzero()
        b_form_tag_ids = b_gold_tag_ids[b_gold_form_mask_ids[:, 0], b_gold_form_mask_ids[:, 1], b_gold_form_mask_ids[:, 2]]
        b_form_tag_ids = b_form_tag_ids.unsqueeze(0).unsqueeze(2)
        b_form_ids = b_gold_form_ids[b_gold_form_mask_ids[:, 0], b_gold_form_mask_ids[:, 1], b_gold_form_mask_ids[:, 2]]
        b_form_token_ids, b_form_token_lengths = get_form_input_seq(b_form_ids)
        b_form_token_mask = b_form_token_ids[:, :, 0, 0] != 0

        b_scores = model(b_form_token_ids, b_form_token_lengths)
        b_losses = model.loss(b_scores, b_form_tag_ids, b_form_token_mask)

        print_loss += sum(b_losses)
        total_loss += sum(b_losses)
        b_form_pred_tag_ids = model.decode(b_scores)
        b_pred_tag_ids = torch.zeros_like(b_gold_tag_ids)
        for tag_id, mask_ids in zip(b_form_pred_tag_ids.squeeze(0), b_gold_form_mask_ids):
            b_pred_tag_ids[mask_ids[0], mask_ids[1], mask_ids[2]] = tag_id

        b_eval_gold_tag_ids = torch.zeros_like(b_eval_gold_tag_ids)
        for tag_id, mask_ids in zip(b_eval_form_tag_ids.squeeze(0), b_eval_gold_form_mask_ids):
            b_eval_gold_tag_ids[mask_ids[0], mask_ids[1], mask_ids[2]] = tag_id

        # b_form_token_ids = b_form_token_ids.detach().cpu().numpy()
        # b_form_token_mask = b_form_token_mask.detach().cpu().numpy()
        b_token_ids = b_token_ids.detach().cpu().numpy()
        b_token_mask = b_token_mask.detach().cpu().numpy()
        # b_gold_tag_ids = b_gold_tag_ids.detach().cpu().numpy()
        b_eval_gold_tag_ids = b_eval_gold_tag_ids.detach().cpu().numpy()
        b_eval_gold_form_mask = b_eval_gold_form_mask.detach().cpu().numpy()
        # b_eval_token_mask = b_eval_token_mask.detach().cpu().numpy()
        # b_token_mask = b_token_mask.detach().cpu().numpy()
        b_pred_tag_ids = b_pred_tag_ids.detach().cpu().numpy()
        b_gold_form_mask = b_gold_form_mask.detach().cpu().numpy()
        gold_tokens = ds.token_ids_to_tokens(b_token_ids, b_token_mask, data_vocab)
        # gold_token_lattice = to_token_lattice(b_gold_tag_ids, b_token_mask.any(axis=2))
        eval_gold_token_lattice = to_token_lattice(b_eval_gold_tag_ids, b_eval_gold_form_mask.any(axis=2))
        eval_pred_token_lattice = to_token_lattice(b_pred_tag_ids, b_gold_form_mask.any(axis=2))
        # print_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        # total_samples.append((gold_tokens, gold_token_lattice, pred_token_lattice))
        print_samples.append((gold_tokens, eval_gold_token_lattice, eval_pred_token_lattice))
        total_samples.append((gold_tokens, eval_gold_token_lattice, eval_pred_token_lattice))
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
    return total_samples


# torch.autograd.set_detect_anomaly(True)
lr = 1e-3
parameters = list(filter(lambda p: p.requires_grad, tagger.parameters()))
adam = AdamW(parameters, lr=lr)
adam = ModelOptimizer(1, adam, parameters, 5.0)
epochs = 9
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    tagger.train()
    run_data(epoch, 'train-gold', gold_train_data, 320, tagger, adam)
    tagger.eval()
    with torch.no_grad():
        samples = run_data(epoch, 'dev-gold', gold_dev_data, 32, tagger)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-dev-gold.conllu')
        samples = run_data(epoch, 'test-gold', gold_test_data, 32, tagger)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-test-gold.conllu')
        samples = run_data(epoch, 'dev-udpipe', udpipe_dev_data, 32, tagger)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-dev-udpipe.conllu')
        samples = run_data(epoch, 'test-udpipe', udpipe_test_data, 32, tagger)
        ds.save_as_conllu(samples, out_dir_path / f'e{epoch}-test-udpipe.conllu')
