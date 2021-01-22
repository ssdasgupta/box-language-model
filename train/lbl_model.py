import argparse
import torchtext, random, torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import wandb

from trainer.Trainer import Trainer
from trainer.data_utils import get_iter
from train.BaseModule import BaseModule


global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1


parser = argparse.ArgumentParser(description='PyTorch log bilinear Language Model')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--n_gram',type=int, default=4, help='Number of previous words to consider')
parser.add_argument('--embedding_dim', type=int, default=50, help='Word embedding dimensions')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epoch')
parser.add_argument('--sep_output', type=int, default=0, help='Use seperate ouput variable')
parser.add_argument('--diag_context', type=int, default=0, help='Use seperate ouput variable')

args = parser.parse_args()

wandb.init(project="box-language-model",  reinit=True)
wandb.config.update(args)

class LBLModel(BaseModule):
    def __init__(self,
                 TEXT = None,
                 embedding_dim = 100,
                 batch_size = 10,
                 n_gram=30,
                 ):
        super(LBLModel, self).__init__()
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.vocab.itos)
        self.embedding_dim = embedding_dim
        self.embeddings_word = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_bias = nn.Embedding(self.vocab_size, 1)
        self.embedding_bias.weight.data = torch.zeros(self.vocab_size, 1)
        if args.sep_output:
            self.embeddings_word_output = nn.Embedding(self.vocab_size, self.embedding_dim)
        if args.diag_context:
            self.C = nn.Embedding(n_gram, self.embedding_dim)
        else:
            self.C = nn.Embedding(n_gram, self.embedding_dim * self.embedding_dim)
    
    def forward(self, x, train = True):
        """ predict, return hidden state so it can be used to intialize the next hidden state """
        context_word_features = self.embeddings_word(x)
        all_gram_idx = torch.arange(self.n_gram).cuda() if use_cuda else torch.arange(self.n_gram)
        all_vocab_idx = torch.arange(self.vocab_size).cuda() if use_cuda else torch.arange(self.vocab_size)
        position_matrix = self.C(all_gram_idx)
        if not args.diag_context:
            position_matrix = position_matrix.reshape(-1, self.embedding_dim, self.embedding_dim)
            context_features = torch.tensordot(context_word_features, position_matrix)
        else:
            context_features = torch.mean(context_word_features * position_matrix , dim=1)

        if args.sep_output:
            all_word = self.embeddings_word_output(all_vocab_idx)
        else:
            all_word = self.embeddings_word(all_vocab_idx)

        decoded = torch.mm(context_features,  all_word.T) + self.embedding_bias(all_vocab_idx).view(-1)
        logits = F.log_softmax(decoded, dim = 1)       
        return logits

TEXT, train_iter, val_iter, test_iter = get_iter(args.batch_size)
model = LBLModel(TEXT=TEXT, embedding_dim=args.embedding_dim, batch_size=args.batch_size, n_gram=args.n_gram)
if use_cuda: 
    model.cuda()
trainer = Trainer(train_iter = train_iter, val_iter = val_iter, TEXT=TEXT, lr=args.lr, n_gram=args.n_gram)
trainer.train_model(model = model, num_epochs = args.num_epochs, path=wandb.run.dir)
