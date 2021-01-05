import argparse
import torchtext, random, torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from box_wrapper import DeltaBoxTensor
from modules import BoxEmbedding

import numpy as np
from tqdm import tqdm
import wandb


global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1

parser = argparse.ArgumentParser(description='PyTorch Box Language Model')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--n_gram',type=int, default=4, help='Number of previous words to consider')
parser.add_argument('--embedding_dim', type=int, default=50, help='Word embedding dimensions')

args = parser.parse_args()

TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="../data",
              train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                             batch_size=args.batch_size, bptt_len=args.n_gram+1, repeat=False)


wandb.init(project="box-language-model",  reinit=True)
wandb.config.update(args)
# wandb.init(project="box-language-model",  reinit=True)


class Trainer:
    def __init__(self, train_iter, val_iter, n_gram):
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.vocab.itos)
        
    def string_to_batch(self, string):
        relevant_split = string.split() # last two words, ignore ___
        ids = [self.word_to_id(word) for word in relevant_split]
        if use_cuda:
            return Variable(torch.LongTensor(ids)).cuda()
        else:
            return Variable(torch.LongTensor(ids))
        
    def word_to_id(self, word, TEXT = TEXT):
        return TEXT.vocab.stoi[word]
    
    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch, n=self.n_gram)
        position_codes = torch.arange(self.n_gram + 1) * self.vocab_size
        x = Variable(torch.LongTensor([ngram[:-1] for ngram in ngrams])) + position_codes[:-1]
        y = Variable(torch.LongTensor([ngram[-1] for ngram in ngrams]))
        if use_cuda:
            return x.cuda(), y.cuda()
        else:
            return x, y
    
    def collect_batch_ngrams(self, batch, n = 5):
        data = torch.flatten(batch.text.T)
        return [tuple(data[idx:idx + n + 1]) for idx in range(0, len(data) - n)]
    
    def train_model(self, model, num_epochs):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=1e-1)
        criterion = nn.NLLLoss()
        
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
#             hidden = model.init_hidden()
            model.train()

            for batch in tqdm(train_iter):
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
            metric = {'train_ppl': train_ppl, 'val_ppl': val_ppl, 'epoch_loss': np.mean(epoch_loss)}
            wandb.log(metric)
            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(model)
        print('Output saved.')
        
    def validate(self, model):
        criterion = nn.NLLLoss()
        aggregate_loss = []
        for batch in self.val_iter:
            x, y = self.batch_to_input(batch)
            if use_cuda: x, y = x.cuda(), y.cuda()
            y_p = model.forward(x, train = False)
            loss = criterion(y_p, y)
            aggregate_loss.append(loss.data.item())        
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl

class BoxAffineTransform(nn.Module):
    box_types = {
        'DeltaBoxTensor': DeltaBoxTensor,
    }
    def __init__(self,
                 TEXT = TEXT,
                 embedding_dim = 50,
                 batch_size = 10,
                 n_gram=4):
        super(BoxAffineTransform, self).__init__()
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.vocab_size = len(TEXT.vocab.itos)
        self.embedding_dim = embedding_dim
        self.embeddings_word = BoxEmbedding(self.vocab_size * self.n_gram, self.embedding_dim, box_type='DeltaBoxTensor')
        self.embedding_bias = nn.Embedding(self.vocab_size, 1)
        self.embedding_bias.weight.data = torch.zeros(self.vocab_size, 1)

        self.position_delta_weight_tail = nn.Embedding(num_embeddings=n_gram,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.position_delta_bias_tail = nn.Embedding(num_embeddings=n_gram,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.position_min_weight_tail = nn.Embedding(num_embeddings=n_gram,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.position_min_bias_tail = nn.Embedding(num_embeddings=n_gram,
                                              embedding_dim=embedding_dim,
                                              sparse= False)

    def position_transformation(self, box, position):
        weight_delta = self.position_delta_weight_tail(position)
        weight_min = self.position_min_weight_tail(position)
        bias_delta = self.position_delta_bias_tail(position)
        bias_min = self.position_min_bias_tail(position)
        box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min + bias_min
        box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return box

    
    def forward(self, x, train = True):
        context_word_boxes = self.embeddings_word(x)
        all_gram_idx = torch.arange(self.n_gram).cuda() if use_cuda else torch.arange(self.n_gram)
        all_vocab_idx = torch.arange(self.vocab_size).cuda() if use_cuda else torch.arange(self.vocab_size)
        
        transformed_boxes = self.position_transformation(context_word_boxes, all_gram_idx)
        transformed_boxes.data = torch.mean(transformed_boxes.data, dim=1).view(-1,1,2,self.embedding_dim)    
        
        all_word = self.embeddings_word(all_vocab_idx * self.n_gram)
        all_word.data = all_word.data.view(1, self.vocab_size, 2, self.embedding_dim)
        
        dec = all_word.intersection_log_soft_volume(context_word_boxes)
        decoded = dec + self.embedding_bias(all_vocab_idx).view(-1)
        logits = F.log_softmax(decoded, dim = 1)       
        return logits


model = BoxAffineTransform(embedding_dim=args.embedding_dim, batch_size=args.batch_size, n_gram=args.n_gram)
if use_cuda:
    model.cuda()
trainer = Trainer(train_iter = train_iter, val_iter = val_iter, n_gram=args.n_gram)
trainer.train_model(model = model, num_epochs = 40)
