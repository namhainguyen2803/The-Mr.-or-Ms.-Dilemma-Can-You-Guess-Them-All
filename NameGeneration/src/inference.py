"""
    Created by @namhainguyen2803 in 10/05/2023
"""
import main
import torch
import preprocess
CHECKPOINT_PATH = "C:/Users/Admin/The-Mr.-or-Ms.-Dilemma-Can-You-Guess-Them-All/NameGeneration/checkpoint/rnn_params.pt"

# Define model and optimizer

checkpoint = torch.load(CHECKPOINT_PATH)
main.rnn.load_state_dict(checkpoint["model_state_dict"])
main.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]

def create_name(list_string):
    string_name = "Nguyen "
    for i in range(len(list_string)):
        if list_string[i] != "UNK":
            if i == 0 and list_string[i].isalpha() == True:
                string_name += list_string[i].upper()
            elif list_string[i-1] == " " and list_string[i].isalpha() == True:
                string_name += list_string[i].upper()
            else:
                string_name += list_string[i]
    return string_name

for i in range(10):
    lst = preprocess.decode_word(main.rnn.sample())
    print(create_name(lst))