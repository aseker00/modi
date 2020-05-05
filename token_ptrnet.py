import random
from models import *
from models_ptrnet import *
import seqtag_dataset as ds
import seqtag_treebank as tb
from model_utils import *
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
# import sys

root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
# sys.path.insert(0, str(root_dir_path))

partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
vocab, lattices_dataset, gold_dataset = ds.load_morpheme_dataset(root_dir_path, partition)
char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_morpheme_ft_emb(root_dir_path, ft_root_dir_path, vocab)

train_dataset = gold_dataset['train']
dev_dataset = gold_dataset['dev']
test_dataset = gold_dataset['test']
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

device = None
num_tags = len(vocab['tags'])
num_feats = len(vocab['feats'])
tag2id = {v: i for i, v in enumerate(vocab['tags'])}
feats2id = {v: i for i, v in enumerate(vocab['feats'])}
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=100, padding_idx=0)
feats_emb = nn.Embedding(num_embeddings=num_feats, embedding_dim=100, padding_idx=0)
token_char_emb = TokenCharRNNEmbedding(char_ft_emb, 300, 1, 0.0)
token_emb = TokenEmbedding(token_ft_emb, token_char_emb, 0.0)
analysis_emb = AnalysisEmbedding(form_ft_emb, lemma_ft_emb, tag_emb, feats_emb)
analysis_encoder = nn.LSTM(input_size=token_emb.embedding_dim, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
analysis_decoder = nn.LSTM(analysis_emb.embedding_dim + token_emb.embedding_dim, analysis_encoder.hidden_size * 2, 1, dropout=0.0, batch_first=True)
analysis_attention = Attention
ptrnet = AnalysisPtrNet(token_emb, 0.0, analysis_emb, analysis_encoder, 0.0, analysis_decoder, 0.0, analysis_attention)
if device is not None:
    ptrnet.to(device)


def run_batch(batch, model, optimizer, teacher_forcing):
    batch = tuple(t.to(device) for t in batch)
    tokens = batch[0]
    token_lengths = batch[1]
    chars = batch[2]
    char_lengths = batch[3]
    morphemes = batch[4]
    tokens, chars, char_lengths, token_lengths, morphemes = batch_narrow(tokens, chars, char_lengths, token_lengths, morphemes)
    use_teacher_forcing = False
    if optimizer:
        if teacher_forcing is not None and random.uniform(0, 1) < teacher_forcing:
            use_teacher_forcing = True

    loss = model(tokens, chars, char_lengths, token_lengths, morphemes, use_teacher_forcing)
    if optimizer:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def run_epoch(data, model, optimizer=None, teacher_forcing=None):
    for i, batch in enumerate(data):
        run_batch(batch, model, optimizer, teacher_forcing)


adam = AdamW(ptrnet.parameters(), lr=1e-3)
run_epoch(train_dataloader, ptrnet, adam)
