import gzip
from pathlib import Path
import torch
import torch.nn as nn
import fasttext


def _save_to(path: Path, lines: list):
    if path.suffix == ".gz":
        with gzip.open(str(path), "wt") as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    else:
        with open(str(path), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')


def _load_from(path: Path) -> list:
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            lines = f.readlines()
    else:
        with open(str(path), 'r') as f:
            lines = f.readlines()
    return lines


def _save_vec_file(path: Path, words: list, vectors: list):
    lines = []
    for word, vector in zip(words, vectors):
        vector_str = ' '.join([str(num) for num in vector])
        line = f"{word} {vector_str}"
        lines.append(line)
    _save_to(path, lines)


def _load_vec_file(path: Path) -> (dict, list):
    word_indices = {}
    vectors = []
    lines = _load_from(path)
    for i, line in enumerate(lines):
        parts = line.split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        vectors.append(vector)
        word_indices[word] = i
    return word_indices, vectors


ft_model = None


def load_embedding_weight_matrix(model_file_path, vec_file_path: Path, words: list, device=None) -> nn.Embedding:
    if vec_file_path.exists():
        print(f"Loading FastText vectors from {vec_file_path}")
        indices, vectors = _load_vec_file(vec_file_path)
        new_words = [word for word in words if word not in indices]
        word_vectors = [vectors[indices[word]] for word in words if word in indices]
        if len(new_words) > 0:
            new_word_vectors = get_word_vectors(model_file_path, new_words)
            word_vectors.extend(new_word_vectors)
    else:
        word_vectors = get_word_vectors(model_file_path, words)
        if words[0] == '<PAD>':
            word_vectors[0] = [0] * len(word_vectors[0])
        _save_vec_file(vec_file_path, words, word_vectors)
    word_matrix = torch.tensor(word_vectors, dtype=torch.float, device=device)
    return nn.Embedding.from_pretrained(word_matrix, freeze=False, padding_idx=0)


def get_word_vectors(model_path: Path, words: list) -> list:
    global ft_model
    if not ft_model:
        print(f"Loading FastText model from {model_path}")
        ft_model = fasttext.load_model(f'{model_path}')
    return [ft_model.get_word_vector(word) for word in words]
