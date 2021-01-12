import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1

class Trainer:
    def __init__(self, train_iter, val_iter, TEXT, lr=0.001, n_gram=4):
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.n_gram = n_gram
        self.lr = lr
        self.vocab_size = len(TEXT.vocab.itos)

    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch, n=self.n_gram + 1)
        data = Variable(torch.LongTensor(list(ngrams)))
        x = data[:, :-1]
        y = data[:, -1]
        if use_cuda:
            return x.cuda(), y.cuda()
        else:
            return x, y
    
    def collect_batch_ngrams(self, batch, n):
        n = max(1, int(n))
        data = batch.text.view(-1).data.tolist()
        for idx in range(0, len(data)-n+1):
            yield tuple(data[idx:idx+n])
    
    def train_model(self, model, num_epochs):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=self.lr)
        criterion = nn.NLLLoss()
        best_val_ppl = float('inf')
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
#             hidden = model.init_hidden()
            model.train()
            for batch in tqdm(self.train_iter):
                x, y = self.batch_to_input(batch)
                if use_cuda: x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()

                y_pred = model.forward(x, train = True)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.data.item())
            model.eval()
            train_ppl = np.exp(np.mean(epoch_loss))
            val_ppl = self.validate(model)
            best_val_ppl = min(val_ppl, best_val_ppl)
            # val_ppl = 0
            metric = {'train_ppl': train_ppl, 'val_ppl': val_ppl, 'epoch_loss': np.mean(epoch_loss), 'best_val_ppl': best_val_ppl}
            wandb.log(metric)
            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(model)
        print('Output saved.')
        
    def validate(self, model):
        criterion = nn.NLLLoss()
#         hidden = model.init_hidden()
        aggregate_loss = []
        for batch in self.val_iter:
            x, y = self.batch_to_input(batch)
            if use_cuda: x, y = x.cuda(), y.cuda()
            y_p = model.forward(x, train = False)
            loss = criterion(y_p, y)
            aggregate_loss.append(loss.data.item())        
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl

class TrainerFullParam(Trainer):
    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch, n=self.n_gram + 1)
        position_codes = torch.arange(self.n_gram + 1) * self.vocab_size
        data = Variable(torch.LongTensor(list(ngrams)))
        x = data[:, :-1] + position_codes[:-1]
        y = data[:, -1]
        if use_cuda:
            return x.cuda(), y.cuda()
        else:
            return x, y
