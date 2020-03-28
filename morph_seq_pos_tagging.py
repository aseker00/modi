from collections import defaultdict
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from models import *
import dataset as ds
import treebank as tb
from pos_tagging_utils import *
from pathlib import Path
# import sys

root_dir_path = Path.home() / 'dev/aseker00/mogy'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
# sys.path.insert(0, str(root_dir_path))

partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
morpheme_vocab, morpheme_dataset = ds.load_morpheme_dataset(root_dir_path, partition)
char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_ft_emb(root_dir_path, ft_root_dir_path, morpheme_vocab)

train_dataset = morpheme_dataset['train']
dev_dataset = morpheme_dataset['dev']
test_dataset = morpheme_dataset['test']
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

device = None
num_tags = len(morpheme_vocab['tags'])
tag2id = {v: i for i, v in enumerate(morpheme_vocab['tags'])}
token_char_emb = TokenCharRNNEmbedding(char_ft_emb, 300, 1, 0.0)
token_emb = TokenEmbedding(token_ft_emb, token_char_emb, 0.7)
token_encoder = TokenRNN(token_emb.embedding_dim, 300, 1, 0.0)
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=100, padding_idx=0)
decoder_tag_rnn = nn.LSTM(tag_emb.embedding_dim, token_encoder.hidden_size, 1, dropout=0.0, batch_first=True)
decoder_tag_token_rnn = nn.LSTM(tag_emb.embedding_dim + token_emb.embedding_dim, token_encoder.hidden_size, 1, dropout=0.0, batch_first=True)
morph_decoder = MorphemeDecoder(decoder_tag_token_rnn, 0.0, num_tags, tag2id['<EOT>'])
model = Seq2SeqClassifier(token_emb, token_encoder, tag_emb, morph_decoder, device)
if device:
    model.to(device)


def run_batch(batch, tagger, optimizer, teacher_forcing):
    batch = tuple(t.to(device) for t in batch)
    batch_token_seq = batch[0]
    batch_token_lengths = batch[1]
    batch_char_seq = batch[2]
    batch_char_lengths = batch[3]
    batch_morph_seq = batch[4]

    # tokens
    max_token_seq = batch_token_lengths.max()
    token_seq = batch_token_seq[:, :max_token_seq].contiguous()

    # chars
    max_char_seq = batch_char_lengths.max()
    token_char_seq = batch_char_seq[:, :max_token_seq, :max_char_seq].contiguous()
    token_char_lengths = batch_char_lengths[:, :max_token_seq].contiguous()

    # token tags
    num_token_tags = batch_morph_seq.shape[2] // 4
    # append an extra zero pad to the tags - so even if there are 5 morphemes (5 tags) then we can still append the ET
    gold_token_tag_seq = F.pad(batch_morph_seq[:, :max_token_seq, (2 * num_token_tags):(3 * num_token_tags)].contiguous(), (0, 1))
    # Mark lattice ET
    gold_token_tag_seq_et_mask = gold_token_tag_seq != 0
    for token_tags, mask_idx in zip(gold_token_tag_seq, gold_token_tag_seq_et_mask.sum(dim=2)):
        for i in range(token_tags.shape[0]):
            token_tags[i][mask_idx[i]] = tag2id['<EOT>']
    gold_token_tag_seq_mask = gold_token_tag_seq != 0
    gold_tag_seq = [token_tags[:num_tokens][mask[:num_tokens]] for token_tags, mask, num_tokens in zip(gold_token_tag_seq, gold_token_tag_seq_mask, batch_token_lengths)]
    gold_tag_seq = torch.nn.utils.rnn.pad_sequence(gold_tag_seq, batch_first=True)
    gold_tag_seq_mask = gold_tag_seq != 0
    if optimizer:
        if teacher_forcing is not None and random.uniform(0, 1) < teacher_forcing:
            decoded_tag_scores = tagger(token_seq, token_char_seq, token_char_lengths, batch_token_lengths, num_token_tags, gold_tag_seq, True)
        else:
            decoded_tag_scores = tagger(token_seq, token_char_seq, token_char_lengths, batch_token_lengths, num_token_tags, gold_tag_seq, False)
    else:
        decoded_tag_scores = tagger(token_seq, token_char_seq, token_char_lengths, batch_token_lengths, num_token_tags)
    # morpheme tag level sequence loss - align scores and gold tags and mask
    if decoded_tag_scores.shape[1] < gold_tag_seq.shape[1]:
        fill_len = gold_tag_seq.shape[1] - decoded_tag_scores.shape[1]
        decoded_tag_scores = F.pad(decoded_tag_scores, (0, 0, 0, fill_len))
    elif gold_tag_seq.shape[1] < decoded_tag_scores.shape[1]:
        fill_len = decoded_tag_scores.shape[1] - gold_tag_seq.shape[1]
        gold_tag_seq = F.pad(gold_tag_seq, (0, fill_len))
        gold_tag_seq_mask = F.pad(gold_tag_seq_mask, (0, fill_len))
    tag_loss = tagger.loss(decoded_tag_scores, gold_tag_seq, gold_tag_seq_mask)
    # TODO: compute crf loss


    if optimizer:
        tag_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return decoded_tag_scores, gold_tag_seq, gold_tag_seq_mask, tag_loss, token_seq, batch_token_lengths, num_token_tags


def run_epoch(epoch, phase, print_every, data, tagger, optimizer=None, teacher_forcing=None, max_num_batches=None):
    print_loss, epoch_loss = 0, 0
    print_samples, epoch_samples = [], []
    for i, batch in enumerate(data):
        step = i + 1
        decoded_tag_scores, gold_tag_seq, gold_tag_seq_mask, decoded_tag_loss, token_seq, token_seq_lengths, num_token_tags = run_batch(batch, tagger, optimizer, teacher_forcing)
        with torch.no_grad():
            decoded_tag_seq = tagger.decode(decoded_tag_scores)

        # build the decoded tag sequence mask
        decoded_et_tag_seq_mask = decoded_tag_seq == tagger.decoder.et_tag_id
        decoded_et_tag_idx = defaultdict(list)
        for b, idx in [(b.item(), idx.item()) for b, idx in decoded_et_tag_seq_mask.nonzero()]:
            decoded_et_tag_idx[b].append(idx)
        decoded_tag_seq_mask = []
        for b, l in enumerate(token_seq_lengths):
            idx = decoded_tag_seq.shape[1] - 1
            if b in decoded_et_tag_idx:
                if l-1 < len(decoded_et_tag_idx[b]):
                    idx = decoded_et_tag_idx[b][l-1]
            decoded_tag_seq_mask.append(torch.arange(decoded_tag_seq.shape[1]) <= idx)
        decoded_tag_seq_mask = torch.stack(decoded_tag_seq_mask, dim=0)

        # Align tag sequence to token sequence - 6 tags per token (5 tags + et)
        gold_tag_idx = index_token_tag_seq(gold_tag_seq, num_token_tags + 1, tag2id)
        decoded_tag_idx = index_token_tag_seq(decoded_tag_seq, num_token_tags + 1, tag2id)
        gold_tag_idx = torch.tensor(gold_tag_idx)
        decoded_tag_idx = torch.tensor(decoded_tag_idx)
        gold_token_tags = align_token_tags(gold_tag_seq, gold_tag_idx, gold_tag_seq_mask, token_seq_lengths, num_token_tags, tag2id)
        decoded_token_tags = align_token_tags(decoded_tag_seq, decoded_tag_idx, decoded_tag_seq_mask, token_seq_lengths, num_token_tags, tag2id)
        decoded_token_tags = torch.nn.utils.rnn.pad_sequence(decoded_token_tags, batch_first=True)
        gold_token_tags = torch.nn.utils.rnn.pad_sequence(gold_token_tags, batch_first=True)
        decoded_token_tags_mask = decoded_token_tags != 0
        gold_token_tags_mask = gold_token_tags != 0
        samples = to_samples(decoded_token_tags, gold_token_tags, decoded_token_tags_mask, gold_token_tags_mask, token_seq, token_seq_lengths, morpheme_vocab)
        # samples = to_samples(decoded_tag_seq, gold_tag_seq, decoded_tag_seq_mask, gold_tag_seq_mask, token_seq, token_seq_lengths, morpheme_vocab)

        print_samples.append(samples)
        epoch_samples.append(samples)
        print_loss += decoded_tag_loss
        epoch_loss += decoded_tag_loss
        if step % print_every == 0:
            print(f'{phase} epoch {epoch} step {step} loss: {print_loss / print_every}')
            print_sample(print_samples[-1][0][-1], print_samples[-1][1][-1])
            print_loss = 0
            print_samples = []
        if max_num_batches and step == max_num_batches:
            break
    print(f'{phase} epoch {epoch} total loss: {epoch_loss / len(data)}')
    print_scores(epoch_samples, ['_', '<PAD>', '<EOT>'])


for lr in [1e-3]:
    adam = AdamW(model.parameters(), lr=lr)
    for epoch in range(15):
        model.train()
        run_epoch(epoch, 'train', 1, train_dataloader, model, adam, 1.0)
        with torch.no_grad():
            model.eval()
            run_epoch(epoch, 'test', 1, test_dataloader, model)
