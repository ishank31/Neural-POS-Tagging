import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from conllu import parse_incr
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
plt.style.use('ggplot')
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import copy



torch.manual_seed(42)
# Loae the GloVe word vectors
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')


tags = set()
with open('/content/drive/MyDrive/INLP/Assignment - 2/en_atis-ud-train.conllu', 'r', encoding='utf-8') as f:
    for sentence in parse_incr(f):
        for token in sentence:
            # print(f"Word: {token['form']}, POS Tag: {token['upostag']}")
            tags.add(token['upostag'])

def save_vocab(vocab, filename):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)

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

p = 3  
s = 3

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def get_data(path):
  data = []
  data_file = open(path, "r", encoding="utf-8")
  for sentence in parse_incr(data_file):
      sent = sentence.metadata
      tok_sent = sent_tokenize(sent['text'])
      words = []
      tags = []
      for token in sentence:
          # print(f"Word: {token['form']}, POS Tag: {token['upostag']}")
          if token['upostag'] != 'SYM':
            words.append(token['form'])
            tags.append(tag_map[token['upostag']])
      data.append((words, tags))
  return data


train_data = get_data("/content/drive/MyDrive/INLP/Assignment - 2/en_atis-ud-train.conllu")
val_data = get_data("/content/drive/MyDrive/INLP/Assignment - 2/en_atis-ud-dev.conllu")
test_data = get_data("/content/drive/MyDrive/INLP/Assignment - 2/en_atis-ud-test.conllu")

def get_sent_tags(data):
  sentences = []
  tags = []
  for d in data:
    sentences.append(d[0])
    tags.append(d[1])
  return sentences, tags

def create_sliding_windows_with_labels(sentences, labels, p, s, tag_map):
    windows = []
    extended_labels = []

    for sentence, label_sequence in zip(sentences, labels):
        extended_sentence = (['<s>'] * p) + sentence + (['</s>'] * s)
        extended_labels_sequence = ([88] * p) + label_sequence + ([99] * s)  

        for i in range(len(extended_sentence) - p - s):
            window = extended_sentence[i : i + p + s + 1]
            window_labels = extended_labels_sequence[i : i + p + s + 1]
            label = window_labels[p]

            windows.append((window, label))

    return windows


train_sent, train_tags = get_sent_tags(train_data)
val_sent, val_tags = get_sent_tags(val_data)
test_sent, test_tags = get_sent_tags(test_data)

train_result = create_sliding_windows_with_labels(train_sent, train_tags, p, s, tag_map)
val_result = create_sliding_windows_with_labels(val_sent, val_tags, p, s, tag_map)
test_result = create_sliding_windows_with_labels(test_sent, test_tags, p, s, tag_map)


class POSTagDataset(Dataset):
    def __init__(self, data: list[tuple[list[str], int]], num_classes, vocabulary:Vocab|None=None):
        self.sentences = [i[0] for i in data] # list of sentences
        self.labels = [i[1] for i in data]

        # Replace words which occur less than 5 times with <UNK>
        all_words = [word for sentence in self.sentences for word in sentence]

        word_counts = Counter(all_words)
        frequency_threshold = 5

        infrequent_words = {word for word, count in word_counts.items() if count < frequency_threshold}
        sentences_with_unk = [["<unk>" if word in infrequent_words else word for word in sentence]for sentence in self.sentences]

        self.sentences = sentences_with_unk

        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN]) 
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.sentences)

    def __getlabels__(self) -> list[int]:
        return self.labels

    def __getembeddings__(self) -> list[list[float]]:
        return self.embeddings

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the datapoint at `index`."""
        
        return torch.tensor(self.vocabulary.lookup_indices(self.sentences[index])), torch.tensor(self.labels[index])
    

train_dataset = POSTagDataset(train_result,len(tag_map))
val_dataset = POSTagDataset(val_result,len(tag_map), vocabulary=train_dataset.vocabulary)
test_dataset = POSTagDataset(test_result,len(tag_map), vocabulary=train_dataset.vocabulary)

train_vocab = train_dataset.vocabulary.get_stoi()
save_vocab(train_vocab, "ffnn_model_vocab_ps3.pkl")

# dictionary mapping numbers to the words in vocabulary
num_to_word = {value: key for key, value in train_vocab.items()}

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = (p + 1 + s) * 300 # 300 is the embedding dim
hidden_size = 128
num_classes = len(tag_map)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(256, num_classes)
        self.double()

    def forward(self, x):
        x = x.to(torch.double)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
loss_fn = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#-----------------TRAINING AND VALIDATION LOOP----------------

num_epochs = 10
total_step = len(train_loader)
count = 0
true_val_labels = []
predicted_val_labels = []

losses = []
accs = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    total_correct = 0
    total_samples = 0
    val_accs = []
    val_losses = []
    train_accs = []
    train_losses = []
    for i, (sent, label) in enumerate(train_loader):
        num_arr = sent.numpy()

        input_embed = []
        for arr in num_arr:
          embeds = []
          for num in arr:
            word = num_to_word[num]
            embed = glove_vectors[word] if word in glove_vectors.key_to_index else np.random.uniform(-1.0,1.0,size = (300,))

            arr = embed.copy()

            embeds.append(embed)
          embeddings = np.array([item for sublist in embeds for item in sublist])
          input_embed.append(embeddings)

        input_embed = np.array(input_embed, dtype = np.double)
        input_embed = torch.tensor(input_embed, dtype=torch.double)

        input_embed = input_embed.to(device, dtype=torch.double)
        label = label.to(device)


        outputs = model(input_embed)

        loss = loss_fn(outputs, label)

        # # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += label.size(0)
        total_correct += (predicted == label).sum().item()
        train_losses.append(loss.item())
        train_accs.append((total_correct / total_samples) * 100)
        

    #---------VAL-----------
    model.eval()
    total_val_correct = 0
    total_val_samples = 0

    for i, (val_sent, val_label) in enumerate(val_loader):
      num_arr = val_sent.numpy()
      input_embed = []
      for arr in num_arr:
        embeds = []
        for num in arr:
          word = num_to_word[num]
          embed = glove_vectors[word] if word in glove_vectors.key_to_index else np.random.uniform(-1.0,1.0,size = (300,))

          arr = embed.copy()
          embeds.append(embed)
        embeddings = np.array([item for sublist in embeds for item in sublist])
        
        input_embed.append(embeddings)

      input_embed = np.array(input_embed, dtype = np.double)
      input_embed = torch.tensor(input_embed, dtype=torch.double)
      input_embed = input_embed.to(device, dtype=torch.double)
      val_label = val_label.to(device)

      outputs = model(input_embed)
      val_loss = loss_fn(outputs, val_label)
      _, predicted = torch.max(outputs.data, 1)
      total_val_samples += val_label.size(0)
      total_val_correct += (predicted == val_label).sum().item()

      true_val_labels.extend(val_label.cpu().numpy())
      predicted_val_labels.extend(predicted.cpu().numpy())

      val_accs.append((total_val_correct / total_val_samples) * 100)
      val_losses.append(loss.item())


    avg_train_accs = np.array(train_accs).mean()
    avg_val_accs = np.array(val_accs).mean()
    avg_train_loss = np.array(train_losses).mean()
    avg_val_loss = np.array(val_losses).mean()

    losses.append(avg_val_loss)
    accs.append(avg_val_accs)

    # if (i+1) % 100 == 0:
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
                  .format(epoch+1, num_epochs, avg_train_loss, avg_train_accs, avg_val_loss, avg_val_accs))
    



new_loss = []
for l in losses:
  new_loss.append(l.item())


plt.plot(range(len(new_loss)), new_loss)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("FFNN model loss vs Epoch")
plt.show()

plt.plot(range(len(accs)), accs)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("FFNN model accuracy vs Epoch")
plt.show()


print("Validation Accuracy: ", accuracy_score(true_val_labels, predicted_val_labels)*100)
print("Classification Report: \n", classification_report(true_val_labels, predicted_val_labels))

conf_matrix = confusion_matrix(true_val_labels, predicted_val_labels)
class_labels = list(tag_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Validation Confusion Matrix')
plt.show()


#--------------------TESTING LOOP-------------------
true_labels = []
predicted_labels = []

model.eval()
total_val_correct = 0
total_val_samples = 0

with torch.no_grad():
  for i, (test_sent, test_label) in enumerate(test_loader):
        num_arr = test_sent.numpy()
        input_embed = []
        for arr in num_arr:
          embeds = []
          for num in arr:
            word = num_to_word[num]
            embed = glove_vectors[word] if word in glove_vectors.key_to_index else np.random.uniform(-1.0,1.0,size = (300,))

            arr = embed.copy()
            embeds.append(embed)
          embeddings = np.array([item for sublist in embeds for item in sublist])
          input_embed.append(embeddings)

        input_embed = np.array(input_embed, dtype = np.double)
        input_embed = torch.tensor(input_embed, dtype=torch.double)
        input_embed = input_embed.to(device, dtype=torch.double)
        test_label = test_label.to(device)

        outputs = model(input_embed)
        test_loss = loss_fn(outputs, test_label)
        _, predicted = torch.max(outputs.data, 1)
        total_val_samples += test_label.size(0)
        total_val_correct += (predicted == test_label).sum().item()
        true_labels.extend(test_label.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())


print("Test Accuracy: ", accuracy_score(true_labels, predicted_labels)*100)
print("Classification Report: \n", classification_report(true_labels, predicted_labels))

conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_labels = list(tag_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()

torch.save(model, "ffnn_model3_ps3.pt")