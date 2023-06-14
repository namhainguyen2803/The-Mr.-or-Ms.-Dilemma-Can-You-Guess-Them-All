"""
    Created by @namhainguyen2803 in 10/05/2023
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy

import custom_dataset
import model
import preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
LR_RATE = 1e-4
MAX_LENGTH = None
LOAD_MODEL = True
CHECKPOINT_PATH = "C:/Users/Admin/The-Mr.-or-Ms.-Dilemma-Can-You-Guess-Them-All/NameGeneration/checkpoint/rnn_params.pt"
print("Hello World")

real_data = preprocess.retrieve_dataset_from_path("C:/Users/Admin/The-Mr.-or-Ms.-Dilemma-Can-You-Guess-Them-All/NameGeneration/dataset/filter_middle_last_name.csv")
# sample_data = ["nam hai"] * 20000
real_split_character = preprocess.retrieve_character_from_document(real_data)
character_to_token = preprocess.create_character_to_token()
token_to_character = preprocess.DISTINCT_CHARACTER
real_matrix = preprocess.encode_word_from_document(real_split_character, character_to_token, MAX_LENGTH)
trainset = custom_dataset.Dataset(real_matrix)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

# Define model and optimizer
rnn = model.RNN(token_to_character)
optimizer = optim.Adam(rnn.parameters(), lr=LR_RATE)
criterion = nn.CrossEntropyLoss()

if LOAD_MODEL == True:
    checkpoint = torch.load(CHECKPOINT_PATH)
    rnn.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
else:
    start_epoch = 0
best_loss = 1e9

def train_RNN(model, optimizer, start_epoch, MAX_LENGTH):
    if MAX_LENGTH != None:
        for e in range(start_epoch + 1, start_epoch + 1 + NUM_EPOCHS):
            for t, x in enumerate(trainloader):
                model.train()
                x = x.to(DEVICE)
                h_prev = None
                loss = 0
                for i in range(MAX_LENGTH - 1):
                    logits, h_prev = rnn(x[:, i], h_prev)
                    loss += criterion(logits, x[:, i + 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {e}, loss: {loss}")
            print(preprocess.decode_word(model.sample()))
            if e % 2 == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                                     "optimizer_state_dict": optimizer.state_dict(), "epoch": e}
                torch.save(checkpoint, CHECKPOINT_PATH)

    else:
        for e in range(start_epoch + 1, start_epoch + 1 + NUM_EPOCHS):
            for t, x in enumerate(trainloader):
                model.train()
                x = x.to(DEVICE)
                h_prev = None
                loss = 0
                i = 0
                while True:
                    logits, h_prev = rnn(x[:, i], h_prev)
                    loss += criterion(logits, x[:, i + 1])
                    i += 1
                    if x[:, i] == 0:
                        break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {e}, loss: {loss}")
            print(preprocess.decode_word(model.sample()))
            if e % 2 == 0:
                checkpoint_params = {"model_state_dict": model.state_dict(),
                                     "optimizer_state_dict": optimizer.state_dict(), "epoch": e}
                torch.save(checkpoint_params, CHECKPOINT_PATH)

if __name__ == "__main__":
    train_RNN(rnn, optimizer, start_epoch, MAX_LENGTH)

    for i in range(10):
        print(preprocess.decode_word(rnn.sample()))
