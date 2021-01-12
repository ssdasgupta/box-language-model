import torchtext, random, torch

def get_iter(batch_size):
	TEXT = torchtext.data.Field()
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="../data",
              train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                             batch_size=1, bptt_len=args.batch_size, repeat=False)

    return TEXT, train_iter, val_iter, test_iter