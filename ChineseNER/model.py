import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .crf import CRF
import argparse
from .utils import WordVocabulary, LabelVocabulary, get_mask, write_dict
import os


class NamedEntityRecog(nn.Module):
    def __init__(self, word_vocab, label_vocab, word_embed_dim, word_hidden_dim, feature_extractor, tag_num, dropout,
                 pretrain_embed=None, use_crf=False, use_gpu=False):
        super(NamedEntityRecog, self).__init__()
        self.use_crf = use_crf
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

        self.embeds = nn.Embedding(word_vocab.size(), word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(word_vocab.size(), word_embed_dim)))

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, batch_first=True, bidirectional=True)
        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim * 2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim * 2, word_hidden_dim * 2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim * 2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        pretrain_emb[0, :] = np.zeros((1, embedding_dim))
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        else:
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        feature_out = self.hidden2tag(feature_out)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(feature_out, mask)
        else:
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq

    def get_entity_from_sent(self, text):
        self.eval()
        text = list(text)
        text_id = list(map(self.word_vocab.word_to_id, text))
        text_tensor = torch.tensor(text_id).long()
        text_tensor = text_tensor.unsqueeze(0)
        mask = get_mask(text_tensor)
        length = [len(text_id)]
        tag_seq = self.forward(text_tensor, length, mask)
        tag_seq = tag_seq.squeeze(0)
        location = list()
        orgnization = list()
        person = list()
        dict1 = {}

        for word, label in zip(text, tag_seq):
            tag = self.label_vocab.id_to_label(label)
            if tag == 'O':
                continue

            if tag.endswith('LOC'):
                if tag.startswith('B') or tag.startswith('S'):
                    location.append(word)
                else:
                    location[-1] += word

            if tag.endswith('ORG'):
                if tag.startswith('B') or tag.startswith('S'):
                    orgnization.append(word)
                else:
                    orgnization[-1] += word

            if tag.endswith('PER'):
                if tag.startswith('B') or tag.startswith('S'):
                    person.append(word)
                else:
                    person[-1] += word
        if location:
            dict1['location'] = location
        if orgnization:
            dict1['orgnization'] = orgnization
        if person:
            dict1['person'] = person

        return dict1

    def get_entity_from_file(self, input_file, out_file):
        if not input_file.endswith('.txt') or not out_file.endswith('.txt'):
            print('输入文件类型错误')
            return
        with open(input_file, 'r', encoding='utf-8') as f1:
            with open(out_file, 'w', encoding='utf-8') as f2:
                line = f1.readline()
                while line:
                    dict1 = self.get_entity_from_sent(line.strip())
                    if dict1:
                        write_dict(dict1, f2)
                        f2.write('\n')
                    line = f1.readline()


def load():
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--feature_extractor', choices=['lstm', 'cnn'], default='lstm')
    parser.add_argument('--train_path', default='data/msra_train.txt')
    parser.add_argument('--test_path', default='data/msra_test.txt')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--number_normalized', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=True)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    word_vocab = WordVocabulary(args.train_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)

    model = NamedEntityRecog(word_vocab, label_vocab, args.word_embed_dim, args.word_hidden_dim, args.feature_extractor,
                             label_vocab.size(), args.dropout, pretrain_embed=None, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model/lstmTrue')))

    return model
