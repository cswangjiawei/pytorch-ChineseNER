import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


class WordVocabulary(object):

    def __init__(self, train_path, number_normalized):
        self.number_normalized = number_normalized
        self._id_to_word = []
        self._word_to_id = {}
        self._pad = -1
        self._unk = -1
        self.index = 0

        self._id_to_word.append('<PAD>')
        self._word_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1
        self._id_to_word.append('<UNK>')
        self._word_to_id['<UNK>'] = self.index
        self._unk = self.index
        self.index += 1

        with open(os.path.join(os.path.dirname(__file__), train_path), 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk()

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def items(self):
        return self._word_to_id.items()


class LabelVocabulary(object):
    def __init__(self, filename):
        self._id_to_label = []
        self._label_to_id = {}
        self._pad = -1
        self.index = 0

        self._id_to_label.append('<PAD>')
        self._label_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1

        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]

                    if label not in self._label_to_id:
                        self._id_to_label.append(label)
                        self._label_to_id[label] = self.index
                        self.index += 1

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_label)

    def label_to_id(self, label):
        return self._label_to_id[label]

    def id_to_label(self, cur_id):
        return self._id_to_label[cur_id]


def my_collate(batch_tensor):
    word_seq_lengths = torch.LongTensor(list(map(len, batch_tensor)))
    _, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    batch_tensor.sort(key=lambda x: len(x), reverse=True)
    tensor_length = [len(sq) for sq in batch_tensor]
    batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=0)
    return batch_tensor, tensor_length


def my_collate_fn(batch):
    return {key: my_collate([d[key] for d in batch]) for key in batch[0]}


def load_pretrain_emb(embedding_path):
    embedd_dim = 100
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim=100):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    vocab_size = word_vocab.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_vocab.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask


def write_dict(dict1, f):
    for key, val in dict1.items():
        if val:
            for e in val:
                f.write(e+'  ')
