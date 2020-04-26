import random
from sklearn.metrics import classification_report

from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import trange

import lattice_dataset as ds
from lattice_models import *
from seqtag_models import *
from pathlib import Path

root_path = Path.home() / 'dev/aseker00/modi/treebank/spmrl/heb/seqtag'
ft_root_path = Path.home() / 'dev/aseker00/fasttext'
seq_type = 'lattice'
dev_set_path = Path(f'{seq_type}_inf_dev.pth')
test_set_path = Path(f'{seq_type}_inf_test.pth')
train_set_path = Path(f'{seq_type}_inf_train.pth')
char_ft_emb_path = Path('char_ft_emb.pth')
token_ft_emb_path = Path('token_ft_emb.pth')
form_ft_emb_path = Path('form_ft_emb.pth')
lemma_ft_emb_path = Path('lemma_ft_emb.pth')

# if False:
if dev_set_path.exists() and test_set_path.exists() and train_set_path.exists():
    dev_set = torch.load(str(dev_set_path))
    test_set = torch.load(str(test_set_path))
    train_set = torch.load(str(train_set_path))
    vocab = ds.load_vocab(root_path / f'{seq_type}/vocab')
else:
    partition = ['dev', 'test', 'train']
    token_samples, lattice_samples, vocab = ds.load_inf_samples(root_path, partition, seq_type)
    token_lengths = {t: torch.tensor(token_samples[t][1], dtype=torch.long, requires_grad=False)
                     for t in token_samples}
    analysis_lengths = {t: torch.tensor(lattice_samples[t][1], dtype=torch.long, requires_grad=False)
                        for t in lattice_samples}
    token_samples = {t: torch.tensor(token_samples[t][0], dtype=torch.long) for t in token_samples}
    lattice_samples = {t: torch.tensor(lattice_samples[t][0], dtype=torch.long) for t in lattice_samples}
    dev_set = TensorDataset(token_samples['dev'], token_lengths['dev'], lattice_samples['dev'],
                            analysis_lengths['dev'])
    test_set = TensorDataset(token_samples['test'], token_lengths['test'], lattice_samples['test'],
                             analysis_lengths['test'])
    train_set = TensorDataset(token_samples['train'], token_lengths['train'], lattice_samples['train'],
                              analysis_lengths['train'])
    torch.save(dev_set, str(dev_set_path))
    torch.save(test_set, str(test_set_path))
    torch.save(train_set, str(train_set_path))
if (char_ft_emb_path.exists() and token_ft_emb_path.exists() and form_ft_emb_path.exists() and
        lemma_ft_emb_path.exists()):
    char_ft_emb = torch.load(char_ft_emb_path)
    token_ft_emb = torch.load(token_ft_emb_path)
    form_ft_emb = torch.load(form_ft_emb_path)
    lemma_ft_emb = torch.load(lemma_ft_emb_path)
else:
    char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_ft_emb(root_path / f'{seq_type}/vocab',
                                                                          ft_root_path, vocab)
    torch.save(char_ft_emb, str(char_ft_emb_path))
    torch.save(token_ft_emb, str(token_ft_emb_path))
    torch.save(form_ft_emb, str(form_ft_emb_path))
    torch.save(lemma_ft_emb, str(lemma_ft_emb_path))
train_data = DataLoader(train_set, batch_size=1, shuffle=True)
dev_data = DataLoader(dev_set, batch_size=1)
test_data = DataLoader(test_set, batch_size=1)

device = None
num_tags = len(vocab['tags'])
num_feats = len(vocab['feats'])
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=50, padding_idx=0)
feats_emb = nn.Embedding(num_embeddings=num_feats, embedding_dim=50, padding_idx=0)
token_char_emb = TokenCharEmbedding(token_ft_emb, char_ft_emb, 100)
num_morpheme_feats = train_set.tensors[2].shape[-1] - 4
lattice_emb = AnalysisEmbedding(form_ft_emb, lemma_ft_emb, tag_emb, feats_emb, num_morpheme_feats)
lattice_encoder = nn.LSTM(input_size=lattice_emb.embedding_dim, hidden_size=300,
                          num_layers=1, bidirectional=True, batch_first=True, dropout=0.0)
analysis_decoder = nn.LSTM(input_size=token_char_emb.embedding_dim + lattice_emb.embedding_dim,
                           hidden_size=lattice_encoder.hidden_size * 2, num_layers=1, batch_first=True, dropout=0.0)
analysis_attn = SequenceStepAttention()
sos = [vocab['form2id']['<SOS>'], vocab['lemma2id']['<SOS>'],
       vocab['tag2id']['<SOS>']] + [vocab['feats2id']['<SOS>']] * num_morpheme_feats
sos = torch.tensor(sos, dtype=torch.long, device=device)
ptrnet = LatticeTokenPtrNet(lattice_emb, token_char_emb, lattice_encoder, analysis_decoder, analysis_attn, sos)
if device is not None:
    ptrnet.to(device)
print(ptrnet)


class ModelOptimizer:
    def __init__(self, step_every, optimizer, parameters, max_grad_norm):
        self.optimizer = optimizer
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm
        self.step_every = step_every
        self.steps = 0

    def step(self, loss):
        self.steps += 1
        # loss = loss/self.step_every
        loss.backward()
        if self.steps % self.step_every == 0:
            self._step()

    def force_step(self):
        if self.steps % self.step_every > 0:
            self._step()

    def _step(self):
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(parameters=self.parameters, max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()


def print_sample_labels(sample):
    print(f'tokens: {sample[0]}')
    print(f'gold: {sample[1]}')
    print(f'pred: {sample[2]}')


def to_lattice_data(tokens, token_mask, lattice, analysis_indices):
    token_sample = tokens[:, :, 0, 0][token_mask]
    lattice_sample = pack_lattice(lattice, token_mask, analysis_indices)
    return ds.sample_to_data(token_sample.cpu().numpy(), lattice_sample.cpu().numpy(), vocab)


def pack_lattice(lattice, mask, indices):
    morpheme_size = lattice.shape[-1]
    analysis_size = lattice.shape[-2]
    if indices.shape == mask.shape:
        indices = indices[mask]
    else:
        indices = indices.squeeze(0)
    index = indices.unsqueeze(-1).repeat(1, analysis_size).unsqueeze(-1).repeat(1, 1, morpheme_size).unsqueeze(1)
    return torch.gather(lattice[mask], 1, index).squeeze(1)


def to_tags_arr(token_mask, lattice, analysis_indices):
    lattice_sample = pack_lattice(lattice, token_mask, analysis_indices)
    lattice_sample = lattice_sample.cpu().numpy()
    return ds.to_tag_vec(lattice_sample[:, :, 2], vocab)


def to_tokens_arr(tokens, token_mask):
    token_sample = tokens[:, :, 0, 0][token_mask]
    token_sample = token_sample.cpu().numpy()
    return ds.to_token_vec(token_sample, vocab)


def print_label_metrics(samples):
    gold_labels = [tag for sample in samples for tags in sample[1] for tag in tags]
    pred_labels = [tag for sample in samples for tags in sample[2] for tag in tags]
    labels = set(pred_labels + gold_labels)
    labels.discard('<PAD>')
    print(classification_report(gold_labels, pred_labels, labels=list(labels)))
    # print(confusion_matrix(gold_tags, pred_tags))
    # precision, recall, fscore, support = precision_recall_fscore_support(gold_tags, pred_tags)


def run_data(epoch, phase, data, print_every, model, optimizer=None, teacher_forcing=None):
    total_loss, print_loss = 0, 0
    total_labels, print_labels = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_tokens = batch[0]
        b_token_lengths = batch[1]
        b_is_gold = batch[2][:, :, :, 0, 0]
        b_lattice = batch[2][:, :, :, :, 1:]
        b_analysis_lengths = batch[3]
        b_token_mask = b_tokens[:, :, 0, 0] != 0
        b_batch_size = b_lattice.shape[0]
        b_token_seq_size = b_lattice.shape[1]
        b_analysis_seq_size = b_lattice.shape[2]
        b_gold_indices = torch.ones((b_is_gold.shape[0], b_is_gold.shape[1]), dtype=torch.long, device=device,
                                    requires_grad=False) * (-1)
        b_lattice_mask_index = torch.arange(b_analysis_seq_size, dtype=torch.long, device=device,
                                            requires_grad=False).repeat(b_batch_size, b_token_seq_size, 1)
        b_lattice_mask = torch.lt(b_lattice_mask_index, b_analysis_lengths.unsqueeze(dim=2))
        for idx in b_is_gold.nonzero():
            b_gold_indices[idx[0], idx[1]] = idx[2]
        # to_lattice_data(b_tokens, b_token_mask, b_lattice, b_gold_indices)
        teach = optimizer is not None and (teacher_forcing is None or random.uniform(0, 1) < teacher_forcing)
        if teach:
            b_scores = model(b_lattice, b_lattice_mask, b_tokens, b_token_lengths, b_gold_indices)
        else:
            b_scores = model(b_lattice, b_lattice_mask, b_tokens, b_token_lengths)
        # mask = torch.arange(lattice.shape[2]).repeat(1, lattice.shape[1], 1) < analysis_lengths.unsqueeze(dim=2).repeat(1, 1, lattice.shape[2])
        b_loss = model.loss(b_scores, b_gold_indices, b_token_mask)
        print_loss += b_loss
        total_loss += b_loss
        b_pred_indices = model.decode(b_scores)
        pred_labels_arr = to_tags_arr(b_token_mask, b_lattice, b_pred_indices)
        gold_labels_arr = to_tags_arr(b_token_mask, b_lattice, b_gold_indices)
        gold_tokens_arr = to_tokens_arr(b_tokens, b_token_mask)
        print_labels.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        total_labels.append((gold_tokens_arr, gold_labels_arr, pred_labels_arr))
        if optimizer is not None:
            optimizer.step(b_loss)
        if (i + 1) % print_every == 0:
            print(f'epoch {epoch}, {phase} step {i + 1}, loss: {print_loss / print_every}')
            print_label_metrics(print_labels)
            print_sample_labels(print_labels[-1])
            print_loss = 0
            print_labels = []
    if optimizer is not None:
        optimizer.force_step()
    print(f'epoch {epoch}, {phase} total loss: {total_loss / len(data)}')
    print_label_metrics(total_labels)


lr = 1e-3
adam = AdamW(ptrnet.parameters(), lr=lr)
adam = ModelOptimizer(10, adam, ptrnet.parameters(), 5.0)
epochs = 3
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    ptrnet.train()
    run_data(epoch, 'train', train_data, 10, ptrnet, adam, 1.0)
    ptrnet.eval()
    with torch.no_grad():
        run_data(epoch, 'dev-inf', dev_data, 10, ptrnet)
        run_data(epoch, 'test-inf', test_data, 10, ptrnet)
