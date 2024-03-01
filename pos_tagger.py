import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from conllu import parse_incr
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
from torch.utils.data import Dataset
# from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import string
plt.style.use('ggplot')
import argparse 
import pickle

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
P = 3
S = 3

tag_map = {'VERB': 0,
 'ADJ': 1,
 'DET': 2,
 'ADP': 3,
 'PRON': 4,
 'CCONJ': 5,
 'NUM': 6,
 'INTJ': 7,
 'ADV': 8,
 'AUX': 9,
 'PROPN': 10,
 'NOUN': 11,
 'PART': 12}

num_to_tag = {0: 'VERB',
 1: 'ADJ',
 2: 'DET',
 3: 'ADP',
 4: 'PRON',
 5: 'CCONJ',
 6: 'NUM',
 7: 'INTJ',
 8: 'ADV',
 9: 'AUX',
 10: 'PROPN',
 11: 'NOUN',
 12: 'PART'}

def load_vocab(filename):
    with open(filename, 'rb') as file:
        vocab = pickle.load(file)
    return vocab

def create_sliding_windows(sentences, p, s, tag_map):
    windows = []

    extended_sentence = (['<s>'] * p) + sentences + (['</s>'] * s)
            
    for i in range(len(extended_sentence) - p - s):
        window = extended_sentence[i : i + p + s + 1]
        windows.append(window)

    return windows

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim): # vocab size will be the input
        super(BiLSTMPOSTagger, self).__init__()
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 *  hidden_dim, output_dim)

    def forward(self, input: torch.Tensor):
        embeddings = self.embed(input)
        output, _ = self.lstm(embeddings)
        output = self.fc(output)

        return output

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.fc = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(dropout_rate)
        self.double()

    def forward(self, x):
        x = x.to(torch.double)
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = self.relu(out)
        
       
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="type of  model to be used")
    args = parser.parse_args()
    model_type = args.model_type

    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

    lstm_model = torch.load("lstm_model.pt")
    ffnn_model = torch.load("ffnn_model.pt")

    lstm_vocab = load_vocab("lstm_model_vocab.pkl")
    ffnn_vocab = load_vocab("ffnn_model_vocab.pkl")

    INPUT_DIM = len(lstm_vocab) # For LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_seq = input("Enter the sentence: ")

    while input_seq != "exit":

        input_seq = word_tokenize(input_seq)
        input_seq = [word.lower() for word in input_seq]

        cleaned_seq = [word for word in input_seq if word not in string.punctuation]

        if model_type == "f":
            input_seq = create_sliding_windows(cleaned_seq, P, S, tag_map)
            # print("input seq", input_seq)
            input_embed = []
            for seq in input_seq:
                embeds = []
                    
                for word in seq:
                    embed = glove_vectors[word] if word in glove_vectors.key_to_index else np.random.uniform(-1.0,1.0,size = (300,))
                    arr = embed.copy()

                
                    embeds.append(embed)
                embeddings = np.array([item for sublist in embeds for item in sublist])
            
                input_embed.append(embeddings)
            

            input_embed = np.array(input_embed, dtype = np.double)
            # print("input_embed", input_embed.shape)
            input_embed = torch.tensor(input_embed, dtype=torch.double)
            input_embed = input_embed.to(device, dtype=torch.double)
             
                    
            outputs = ffnn_model(input_embed)
            _, predicted = torch.max(outputs.data, 1)
            # print("Predicted", predicted)

            predicted = predicted.numpy()
            # print("Predicted", predicted)
            for word, tag in zip(cleaned_seq, predicted):
                print(f"{word}\t{num_to_tag[tag]}")
                        
        elif model_type == "r":
            encode = []
            for i in cleaned_seq:
                if i in lstm_vocab:
                    encode.append(lstm_vocab[i])
                else:
                    encode.append(lstm_vocab[UNKNOWN_TOKEN])

            max_length = INPUT_DIM
            padded_sequence = pad_sequence([torch.tensor(encode)], batch_first=True, padding_value=lstm_vocab[PAD_TOKEN])
            padded_sequence = padded_sequence[:max_length]

            output = lstm_model(padded_sequence)
            _, predicted = torch.max(output.data, 2)
            predicted = predicted[0].numpy()

            for word, tag in zip(cleaned_seq, predicted):
                print(f"{word}\t{num_to_tag[tag]}")
        
        input_seq = input("Enter the sentence: ")


