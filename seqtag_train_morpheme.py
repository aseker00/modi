from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

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
    token_samples, morph_samples, vocab = ds.load_samples(root_path, partition, seq_type, 'var')
    token_lengths = {t: torch.tensor(token_samples[t][1], dtype=torch.long, requires_grad=False)
                     for t in token_samples}
    token_samples = {t: torch.tensor(token_samples[t][0], dtype=torch.long) for t in token_samples}
    morph_samples = {t: torch.tensor(morph_samples[t], dtype=torch.long) for t in morph_samples}
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
    char_ft_emb, token_ft_emb, _, _ = ds.load_ft_emb(root_path / f'{seq_type}/vocab', ft_root_path, vocab)
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
model = Seq2SeqClassifier(token_char_emb, token_encoder, tag_emb, tag_decoder, max_tag_seq_len, sos, eot)
if device is not None:
    model.to(device)
print(model)

to_tag = lambda x: vocab['tags'][x]
to_token = lambda x: vocab['tokens'][x]
to_tag_vec = np.vectorize(to_tag)
to_token_vec = np.vectorize(to_token)
lr = 1e-2
adam = AdamW(model.parameters(), lr=lr)
# torch.autograd.set_detect_anomaly(True)
for epoch in range(3):
    for i, batch in enumerate(train_data):
        b_tokens = batch[0]
        b_token_lengths = batch[1]
        b_morphemes = batch[2]
        b_tags = b_morphemes[:, :, :, 2]
        b_token_mask = b_tokens[:, :, 0, 0] != 0
        b_tags_mask = b_tags != 0
        # [b_max_tokens, b_max_chars] = b_token_lengths[:, :].max(dim=1)[0][0].tolist()
        b_scores = model(b_tokens, b_token_lengths, b_tags)
        b_loss = model.loss(b_scores, b_tags[b_tags_mask])
        b_loss.backward()
        adam.step()
        adam.zero_grad()
        if (i + 1) % 20 == 0:
            print(f'{i + 1}, {b_loss.item()}')
            b_pred = model.decode(b_scores).unsqueeze(dim=0)
            gold_sample = b_tags[-1, b_token_mask[-1], :]
            pred_sample = b_pred[-1, :, :]
            token_sample = b_tokens[-1, b_token_mask[-1], 0, 0]
            print(f'tokens: {to_token_vec(token_sample.cpu().numpy())}')
            print(f'gold: {to_tag_vec(gold_sample.cpu().numpy())}')
            print(f'pred: {to_tag_vec(pred_sample.cpu().numpy())}')
