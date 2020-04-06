import random
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from models import *
import heb_sprml_treebank_dataset as ds
import heb_spmrl_treebank as tb
from pos_tagging_utils import *
from pathlib import Path
# import sys

root_dir_path = Path.home() / 'dev/aseker00/modi'
ft_root_dir_path = Path.home() / 'dev/aseker00/fasttext'
# sys.path.insert(0, str(root_dir_path))

partition = tb.load_lattices(root_dir_path, ['dev', 'test', 'train'])
morpheme_vocab, _, gold_dataset = ds.load_morpheme_dataset(root_dir_path, partition)
char_ft_emb, token_ft_emb, form_ft_emb, lemma_ft_emb = ds.load_morpheme_ft_emb(root_dir_path, ft_root_dir_path, morpheme_vocab)

train_dataset = gold_dataset['train']
dev_dataset = gold_dataset['dev']
test_dataset = gold_dataset['test']
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

device = None
num_tags = len(morpheme_vocab['tags'])
tag2id = {v: i for i, v in enumerate(morpheme_vocab['tags'])}
token_char_emb = TokenCharRNNEmbedding(char_ft_emb, 300, 1, 0.0)
token_emb = TokenEmbedding(token_ft_emb, token_char_emb, 0.0)
# token_encoder = TokenRNN(token_emb.embedding_dim, 300, 1, 0.0)
token_encoder = nn.LSTM(input_size=token_emb.embedding_dim, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
# token_encoder = nn.LSTM(input_size=token_emb.embedding_dim, hidden_size=300, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True)
tag_emb = nn.Embedding(num_embeddings=num_tags, embedding_dim=100, padding_idx=0)
decoder_tag_rnn = nn.LSTM(tag_emb.embedding_dim, token_encoder.hidden_size, 1, dropout=0.0, batch_first=True)
decoder_tag_token_rnn = nn.LSTM(tag_emb.embedding_dim + token_emb.embedding_dim, token_encoder.hidden_size * 2, 1, dropout=0.0, batch_first=True)
morph_decoder = MorphemeDecoder(decoder_tag_token_rnn, 0.0, num_tags, tag2id['<EOT>'])
model = Seq2SeqClassifier(token_emb, 0.0, token_encoder, tag_emb, morph_decoder, device)
if device is not None:
    model.to(device)


def score_tokens(tokens, chars, char_lengths, token_lengths, morphemes, max_morph_per_token, tagger,
                 use_teacher_forcing):
    gold_tags = F.pad(morphemes[:, :, (2 * max_morph_per_token):(3 * max_morph_per_token)], [0, 1])
    tokens, chars, char_lengths, token_lengths, gold_tags = batch_narrow(tokens, chars, char_lengths, token_lengths,
                                                                         gold_tags)
    # tokens = F.pad(tokens, [0, 1])
    # chars = F.pad(chars, [0, 0, 0, 1])
    # char_lengths = F.pad(char_lengths, [0, 1])

    # Append <EOT> labels
    gold_tags_mask = gold_tags != tag2id['<PAD>']
    for sample, mask in zip(gold_tags, gold_tags_mask.sum(dim=2)):
        for i in range(sample.shape[0]):
            sample[i][mask[i]] = tag2id['<EOT>']

    gold_tags_mask = gold_tags != tag2id['<PAD>']
    gold_tags = [sample[:num][mask[:num]] for sample, mask, num in zip(gold_tags, gold_tags_mask, token_lengths)]
    gold_tags = torch.nn.utils.rnn.pad_sequence(gold_tags, batch_first=True)
    # Model forward step
    if use_teacher_forcing:
        scores = tagger(tokens, chars, char_lengths, token_lengths, max_morph_per_token, gold_tags)
    else:
        scores = tagger(tokens, chars, char_lengths, token_lengths, max_morph_per_token)
        # Align decoded scores and gold tags/mask
        scores_len = scores.shape[1]
        gold_len = gold_tags.shape[1]
        fill_len = gold_len - scores_len
        if fill_len > 0:
            scores = F.pad(scores, [0, 0, 0, fill_len])
        elif fill_len < 0:
            gold_tags = F.pad(gold_tags, [0, -fill_len])
    gold_tags_mask = gold_tags != tag2id['<PAD>']
    loss = tagger.decoder.loss(scores, gold_tags, gold_tags_mask)
    return scores, gold_tags, tokens, loss


def score_tags(token_scores, gold_token_tags, tagger):
    token_tags = tagger.decoder.decode(token_scores)
    (tag_scores, pred_tags, pred_tags_indices, pred_tags_mask,
     gold_tags, gold_tags_indices, gold_tags_mask) = batch_seq_filter_mask(token_scores, token_tags, gold_token_tags,
                                                                           tag2id['<EOT>'])
    loss = tagger.loss_crf(tag_scores, gold_tags, gold_tags_mask)
    return tag_scores,pred_tags_indices, pred_tags_mask, gold_tags, gold_tags_indices, gold_tags_mask, loss


def run_batch(batch, tagger, optimizer, teacher_forcing):
    batch = tuple(t.to(device) for t in batch)
    tokens = batch[0]
    token_lengths = batch[1]
    chars = batch[2]
    char_lengths = batch[3]
    morphemes = batch[4]
    max_morph_per_token = morphemes.shape[2] // 4
    use_teacher_forcing = False
    if optimizer:
        if teacher_forcing is not None and random.uniform(0, 1) < teacher_forcing:
            use_teacher_forcing = True
    token_tag_scores, gold_token_tags, tokens, token_loss = score_tokens(tokens, chars, char_lengths, token_lengths,
                                                                         morphemes, max_morph_per_token, tagger,
                                                                         use_teacher_forcing)
    tag_scores, tag_indices, tag_masks, gold_tags, gold_indices, gold_masks, tag_loss = score_tags(token_tag_scores,
                                                                                                   gold_token_tags,
                                                                                                   tagger)
    if optimizer:
        token_loss.backward(retain_graph=True)
        tag_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return tokens, token_lengths, max_morph_per_token, tag_scores, tag_indices, tag_masks, tag_loss, gold_tags, \
           gold_indices, gold_masks


def run_epoch(epoch, phase, print_every, data, tagger, optimizer=None, teacher_forcing=None, max_num_batches=None):
    print_loss, epoch_loss = 0, 0
    print_samples, epoch_samples = [], []
    for i, batch in enumerate(data):
        step = i + 1
        (tokens, token_lengths, max_tags_pre_token,
         scores, pred_tags_indices, pred_tags_masks, pred_tags_loss,
         gold_tags, gold_tags_indices, gold_tags_masks) = run_batch(batch, tagger, optimizer, teacher_forcing)

        # Build decoded samples for printouts and accuracy evaluation
        # Decode
        with torch.no_grad():
            decoded_tags = tagger.decode_crf(scores, pred_tags_masks)
        # Reconstruct token level tags
        gold_token_tags = batch_mask_select_reconstruct(gold_tags, gold_tags_indices, gold_tags_masks, tag2id['<EOT>'])
        decoded_token_tags = batch_mask_select_reconstruct(decoded_tags, pred_tags_indices, pred_tags_masks,
                                                           tag2id['<EOT>'])
        # Expand tag to token sequence - 6 tags per token (5 tags + <EOT>)
        gold_tags_indices = gold_token_tags == tag2id['<EOT>']
        pred_tags_indices = decoded_token_tags == tag2id['<EOT>']
        gold_token_tags = batch_expand_token_labels(gold_token_tags, gold_tags_indices, max_tags_pre_token + 1,
                                                    tag2id['_'])
        decoded_token_tags = batch_expand_token_labels(decoded_token_tags,pred_tags_indices,  max_tags_pre_token + 1,
                                                       tag2id['_'])
        decoded_token_tag_masks = decoded_token_tags != 0
        gold_token_tag_masks = gold_token_tags != 0
        samples = to_samples(decoded_token_tags, gold_token_tags, decoded_token_tag_masks, gold_token_tag_masks,
                             tokens, token_lengths, morpheme_vocab)
        print_samples.append(samples)
        epoch_samples.append(samples)
        print_loss += pred_tags_loss
        epoch_loss += pred_tags_loss
        if step % print_every == 0:
            print(f'{phase} epoch {epoch} step {step} loss: {print_loss / print_every}')
            print_sample(print_samples[-1][0][-1], print_samples[-1][1][-1], ['<PAD>', '<EOT>', '_'])
            print_loss = 0
            print_samples = []
        if max_num_batches and step == max_num_batches:
            break
    print(f'{phase} epoch {epoch} total loss: {epoch_loss / len(data)}')
    print_scores(epoch_samples, ['<PAD>', '<EOT>', '_'])


for lr in [1e-3]:
    adam = AdamW(model.parameters(), lr=lr)
    for epoch in range(15):
        model.train()
        run_epoch(epoch, 'train', 32, train_dataloader, model, adam, 0.0)
        with torch.no_grad():
            model.eval()
            run_epoch(epoch, 'test', 32, test_dataloader, model)
