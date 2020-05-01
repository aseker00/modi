import random
from tqdm import trange
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import lattice_dataset as ds
from lattice_models import *
from seqtag_models import *
from seqtag_utils import *
from pathlib import Path

root_path = Path.home() / 'dev/aseker00/modi'
tb_path = root_path / 'treebank/spmrl/heb'
data_path = root_path / 'data/spmrl/heb'
ft_root_path = Path.home() / 'dev/aseker00/fasttext'
seq_type = 'lattice'
dev_set_path = data_path / seq_type / 'dev-inf.pth'
test_set_path = data_path / seq_type / 'test-inf.pth'
train_set_path = data_path / seq_type / 'train-inf.pth'
# test_set_path = Path(f'{seq_type}_inf_test.pth')
# train_set_path = Path(f'{seq_type}_inf_train.pth')
char_ft_emb_path = data_path / 'char-ft-emb.pth'
token_ft_emb_path = data_path / 'token-ft-emb.pth'
form_ft_emb_path = data_path / 'form-ft-emb.pth'
lemma_ft_emb_path = data_path / 'lemma-ft-emb.pth'

# if False:
if dev_set_path.exists() and test_set_path.exists() and train_set_path.exists():
    dev_set = torch.load(str(dev_set_path))
    test_set = torch.load(str(test_set_path))
    train_set = torch.load(str(train_set_path))
    vocab = ds.load_vocab(tb_path / f'{seq_type}/vocab')
else:
    partition = ['dev', 'test', 'train']
    token_arr, lattice_arr, vocab = ds.load_inf_lattices(tb_path, partition, seq_type)
    token_lengths = {t: torch.tensor(token_arr[t][1], dtype=torch.long, requires_grad=False) for t in token_arr}
    analysis_lengths = {t: torch.tensor(lattice_arr[t][1], dtype=torch.long, requires_grad=False) for t in lattice_arr}
    token_samples = {t: torch.tensor(token_arr[t][0], dtype=torch.long) for t in token_arr}
    lattice_samples = {t: torch.tensor(lattice_arr[t][0], dtype=torch.long) for t in lattice_arr}
    dev_set = TensorDataset(*[s['dev'] for s in [token_samples, token_lengths, lattice_samples, analysis_lengths]])
    test_set = TensorDataset(*[s['test'] for s in [token_samples, token_lengths, lattice_samples, analysis_lengths]])
    train_set = TensorDataset(*[s['train'] for s in [token_samples, token_lengths, lattice_samples, analysis_lengths]])
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
    char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_ft_vec(tb_path / f'{seq_type}/vocab',
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


def pack_lattice(lattice_ids, mask, indices):
    morpheme_size = lattice_ids.shape[-1]
    analysis_size = lattice_ids.shape[-2]
    # TODO: Remove condition once the pred_indices and gold_indices have the same shape [batch_size, token_seq_size]
    if indices.shape == mask.shape:
        # Gold indices (shape is [batch_size, token_seq_size], so needs to be masked)
        indices = indices[mask]
    else:
        # Pred indices (already masked by the model)
        indices = indices.squeeze(0)
    index = indices.unsqueeze(-1).repeat(1, analysis_size).unsqueeze(-1).repeat(1, 1, morpheme_size).unsqueeze(1)
    # gather: [n, a, m, 10] -> [n, 1, m, 10]
    # n - token seq len
    # a - analysis seq len
    # m - morphemes per analysis
    # 10 - morpheme size (form, lemma, tag, 7 features)
    return torch.gather(lattice_ids[mask], 1, index).squeeze(1)


def to_token_lattice(lattice_ids, token_mask, analysis_indices):
    token_lattice_ids = pack_lattice(lattice_ids, token_mask, analysis_indices)
    return ds.to_token_lattice(token_lattice_ids.cpu().numpy(), vocab)


def run_data(epoch, phase, data, print_every, model, optimizer=None, teacher_forcing=None):
    total_loss, print_loss = 0, 0
    total_samples, print_samples = [], []
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        b_token_ids = batch[0]
        b_token_lengths = batch[1]
        b_is_gold = batch[2][:, :, :, 0, 0]
        b_lattice_ids = batch[2][:, :, :, :, 1:]
        b_analysis_lengths = batch[3]
        b_token_mask = b_token_ids[:, :, 0, 0] != 0
        b_batch_size = b_lattice_ids.shape[0]
        b_token_seq_size = b_lattice_ids.shape[1]
        b_analysis_seq_size = b_lattice_ids.shape[2]
        b_gold_indices = torch.ones((b_is_gold.shape[0], b_is_gold.shape[1]), dtype=torch.long, device=device,
                                    requires_grad=False) * (-1)
        b_lattice_mask_index = torch.arange(b_analysis_seq_size, dtype=torch.long, device=device,
                                            requires_grad=False).repeat(b_batch_size, b_token_seq_size, 1)
        b_lattice_mask = torch.lt(b_lattice_mask_index, b_analysis_lengths.unsqueeze(dim=2))
        for idx in b_is_gold.nonzero():
            b_gold_indices[idx[0], idx[1]] = idx[2]
        teach = optimizer is not None and (teacher_forcing is None or random.uniform(0, 1) < teacher_forcing)
        # TODO: Change model to return scores in the same shape as tokens [batch_size, tokens_seq_size]?
        if teach:
            b_scores = model(b_lattice_ids, b_lattice_mask, b_token_ids, b_token_lengths, b_gold_indices)
        else:
            b_scores = model(b_lattice_ids, b_lattice_mask, b_token_ids, b_token_lengths)
        b_loss = model.loss(b_scores, b_gold_indices, b_token_mask)
        print_loss += b_loss
        total_loss += b_loss
        b_pred_indices = model.decode(b_scores)
        b_token_ids = b_token_ids.cpu().numpy()
        # b_token_mask = b_token_mask.cpu().numpy()
        gold_tokens = ds.to_tokens(b_token_ids, b_token_mask.cpu().numpy(), vocab)
        # b_lattice = b_lattice.cpu().numpy()
        # b_gold_indices = b_gold_indices.cpu().numpy()
        # b_pred_indices = b_pred_indices.cpu().numpy()
        gold_token_lattice = to_token_lattice(b_lattice_ids, b_token_mask, b_gold_indices)
        pred_token_lattice = to_token_lattice(b_lattice_ids, b_token_mask, b_pred_indices)
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
