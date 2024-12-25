from io import open
import glob
import os
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn

from dataloader import n_letters, getAllCategories, randomTrainingExample, lineToTensor, categoryFromOutput
from model import RNN
from utils import timeSince

def train_iter(rnn, criterion, learning_rate, category_tensor, line_tensor, device):
    hidden = rnn.initHidden()
    hidden = hidden.to(device)

    rnn.zero_grad()
    line_tensor = line_tensor.to(device)

    for i in range(line_tensor.size()[0]):
        input_tensor = line_tensor[i]
        input_tensor = input_tensor.to(device)
        output, hidden = rnn(input_tensor, hidden)

    category_tensor = category_tensor.to(device)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item(), rnn

# Just return an output given a line
def evaluate(rnn, line_tensor, device):
    hidden = rnn.initHidden()
    hidden = hidden.to(device)

    line_tensor = line_tensor.to(device)

    for i in range(line_tensor.size()[0]):
        input_tensor = line_tensor[i]
        input_tensor = input_tensor.to(device)
        output, hidden = rnn(input_tensor, hidden)

    return output

def main():
    all_categories, category_lines = getAllCategories()
    n_categories = len(all_categories)

    device = torch.device("cuda")
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    rnn.to(device)

    criterion = nn.NLLLoss()

    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    n_iters = 300000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output, loss, rnn = train_iter(rnn, criterion, learning_rate, category_tensor, line_tensor, device)
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.savefig("loss.png", bbox_inches = 'tight')

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output = evaluate(rnn, line_tensor, device)
        guess, guess_i = categoryFromOutput(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig("confusion.png", bbox_inches = 'tight')

    torch.save(rnn.state_dict(), "rnn.pt")


if __name__ == "__main__":
    main()