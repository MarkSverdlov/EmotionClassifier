import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Adapter:
    def __init__(self, tokens):
        self.codebook = {t: i for (i, t) in enumerate(list(tokens))}
        self.padding = "<PAD>"
        self.unknown = "<UNKNOWN>"

    def tokenize(self, text):
        return [''.join([c.lower() for c in word if c.isalpha()])
                if ''.join([c.lower() for c in word if c.isalpha()]) in self.codebook
                else self.unknown
                for word in text.split()]

    def distort(self, seq, p=0.01):
        choice = np.random.binomial(size=len(seq), n=1, p=p)
        distorted = seq.copy()
        for i, _ in enumerate(seq):
            if choice[i] == 1:
                distorted[i] = self.unknown
        return distorted

    def pad(self, seq, length):
        current_length = len(seq)
        padding_length = length - current_length
        return seq + [self.padding] * padding_length

    def tensor(self, seqs):
        if not isinstance(seqs[0], list):
            seqs = [seqs]
        array = [[self.codebook[t] for t in seq] for seq in seqs]
        return torch.tensor(array, dtype=torch.int32)

    def process(self, seqs, length, p):
        if not isinstance(seqs[0], list):
            seqs = [seqs]
        lens = [len(seq) for seq in seqs]
        return self.tensor([self.pad(self.distort(seq, p=p), length=length) for seq in seqs]), lens


class SentencesDataset(Dataset):
    def __init__(self, data, adapter, length, p):
        self.X = data['tokens']
        self.Y = data['label']
        self.index = data.index
        self.adapter = adapter
        self.length = length
        self.p = p

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x, lens = self.adapter.process(self.X[idx], self.length, self.p)
        x = x.squeeze()
        return torch.cat([x, torch.tensor(lens)]), self.Y[idx]


class TrainingData:
    def __init__(self):
        self.training_loss = {}
        self.validation_loss = {}
        self.epochs = 0

    def plot(self, ax):
        ax.plot(self.training_loss.keys(), self.training_loss.values(), color="blue", label="Training Loss")
        ax.plot(self.validation_loss.keys(), self.validation_loss.values(), color="red", label="Validation Loss")
        ax.set_title("Training and Validation Loss During Training")
        ax.legend()
        ax.setxlim(0, self.epochs)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    try:
        return loss.item()
    except AttributeError:
        return loss


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


if __name__ == "__main__":
    data = pd.read_csv('emotions.csv')
    data['tokens'] = data['text'].transform(lambda x: x.split())
    tokens = set()
    for _, values in data['tokens'].items():
        tokens.update(set(values))
    tokens.update({'<PAD>', '<UNKNOWN>'})
    adapter = Adapter(tokens)
    loader = DataLoader(SentencesDataset(data, adapter, 200, 1e-2), 64, True)
