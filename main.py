import random
import torch
import numpy as np
import argparse
import os
from ChineseNER.utils import WordVocabulary, LabelVocabulary, my_collate_fn, lr_decay
import time
from ChineseNER.dataset import MyDataset
from torch.utils.data import DataLoader
from ChineseNER.model import NamedEntityRecog
import torch.optim as optim
from tensorboardX import SummaryWriter
from ChineseNER.train import train_model, evaluate

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--savedir', default='ChineseNER/model/')
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
    print('use_crf:', args.use_crf)
    print('use_crf_type:', type(args.use_crf))

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    eval_path = "ChineseNER/evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'

    model_name = args.savedir + '/' + args.feature_extractor + str(args.use_crf)
    word_vocab = WordVocabulary(args.train_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)

    # emb_begin = time.time()
    # pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    # emb_end = time.time()
    # emb_min = (emb_end - emb_begin) % 3600 // 60
    # print('build pretrain embed cost {}m'.format(emb_min))

    train_dataset = MyDataset(args.train_path, word_vocab, label_vocab, args.number_normalized)
    test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(word_vocab, label_vocab, args.word_embed_dim, args.word_hidden_dim, args.feature_extractor,
                             label_vocab.size(), args.dropout, pretrain_embed=None, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_begin = time.time()
    print('train begin', '-' * 50)
    print()
    print()

    writer = SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate(test_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on test set:', best_f1)
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-' * 50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    print('-' * 50)
