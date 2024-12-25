import argparse

import torch

from dataloader import n_letters, getAllCategories, lineToTensor, categoryFromOutput
from model import RNN

def preprocess_input(line):
    line_tensor = lineToTensor(line)
    return line_tensor

def main(args):
    all_categories, category_lines = getAllCategories()
    n_categories = len(all_categories)
    device = torch.device("cuda")
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    rnn.load_state_dict(torch.load("rnn.pt"))
    rnn = rnn.to(device)

    line = args.input_name
    line_tensor = preprocess_input(line)

    with torch.no_grad():
        line_tensor = line_tensor.to(device)
        hidden = rnn.initHidden()
        hidden = hidden.to(device)
        for i in range(line_tensor.size()[0]):
            input_tensor = line_tensor[i]
            input_tensor = input_tensor.to(device)
            output, hidden = rnn(input_tensor, hidden)
 
        guess, _ = categoryFromOutput(output, all_categories)

    print(f"Name `{line}` is from `{guess}`")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_name", required=True, type=str)
    args = parser.parse_args()
    main(args)