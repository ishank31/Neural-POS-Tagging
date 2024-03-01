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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pickle
from tqdm import tqdm



torch.manual_seed(42)

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

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def save_vocab(vocab, filename):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def calculate_accuracy(predictions_list, labels_list, pad_index=77):
    total_correct = 0
    total_non_pad = 0

    for predictions, labels in zip(predictions_list, labels_list):
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)

        non_pad_mask = labels != pad_index
        predictions = predictions[non_pad_mask]
        labels = labels[non_pad_mask]

        total_correct += (predictions == labels).sum().item()
        total_non_pad += len(labels)

    accuracy = total_correct / total_non_pad if total_non_pad > 0 else 0
    return accuracy

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

train_sent, train_tags = get_sent_tags(train_data)
val_sent, val_tags = get_sent_tags(val_data)
test_sent, test_tags = get_sent_tags(test_data)

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
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN]) # use min_freq for handling unkown words better
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

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
      """Given a list of datapoints, batch them together"""
      sentences = [i[0] for i in batch]
      labels = [i[1] for i in batch]
      padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN]) # pad sentences with pad token id
      padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(77)) # pad labels with 77 because pad token cannot be entities

      return padded_sentences, padded_labels
    
train_dataset = POSTagDataset(train_data,len(tag_map))
val_dataset = POSTagDataset(val_data,len(tag_map), vocabulary=train_dataset.vocabulary)
test_dataset = POSTagDataset(test_data,len(tag_map), vocabulary=train_dataset.vocabulary)

vocab = train_dataset.vocabulary.get_stoi()
save_vocab(vocab, "lstm_model_vocab.pkl")

vocab_size = len(train_dataset.vocabulary)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn = train_dataset.collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn = val_dataset.collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn = test_dataset.collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_dim): 
        super(BiLSTMPOSTagger, self).__init__()
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 *  hidden_dim, output_dim)

    def forward(self, input: torch.Tensor):
        embeddings = self.embed(input)
        output, _ = self.lstm(embeddings)
        output = self.fc(output)

        return output
    

INPUT_DIM = vocab_size
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
output_dim = len(tag_map)
print(output_dim)
model = BiLSTMPOSTagger(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, output_dim)

loss_function = nn.CrossEntropyLoss(ignore_index=77)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


#----------------TRAINING AND VALIDATION LOOP-------------------
num_epochs = 20
losses = []
accs = []
total_step = len(train_loader)

true_val_labels = []
predicted_val_labels = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):
  total_correct = 0
  total_samples = 0
  val_accs = []
  val_losses = []
  train_accs = []
  train_losses = []
  for i, (sent, label) in enumerate(train_loader):

    sent = sent.to(device)
    label = label.to(device)

    outputs = model(sent)
    to_loss_outputs = outputs.permute(0, 2, 1)
   
    loss = loss_function(to_loss_outputs, label)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 2)
    total_samples += label.size(0)

    overall_accuracy = calculate_accuracy(predicted, label)

    train_accs.append(overall_accuracy)
    train_losses.append(loss.item())


  for i, (sent, label) in enumerate(val_loader):


    sent = sent.to(device)
    label = label.to(device)

    outputs = model(sent)
    to_loss_outputs = outputs.permute(0, 2, 1)
    loss = loss_function(to_loss_outputs, label)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 2)
    total_samples += label.size(0)
    overall_accuracy = calculate_accuracy(predicted, label)

    true_val_labels.extend(label.cpu().numpy())
    predicted_val_labels.extend(predicted.cpu().numpy())

    val_accs.append(overall_accuracy)
    val_losses.append(loss.item())


  avg_train_loss = np.array(train_losses).mean()
  avg_train_accs = np.array(train_accs).mean()
  avg_val_loss = np.array(val_losses).mean()
  avg_val_accs = np.array(val_accs).mean()
  losses.append(avg_val_loss)
  accs.append(avg_val_accs)
  print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
  .format(epoch+1, num_epochs, avg_train_loss, avg_train_accs*100, avg_val_loss, avg_val_accs*100))



new_loss = []
for l in losses:
  new_loss.append(l.item())

plt.plot(range(len(new_loss)), new_loss)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("LSTM model loss vs Epoch")
plt.show()

accs = [value * 100 for value in accs]
plt.plot(range(len(accs)), accs)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("LSTM model accuracy vs Epoch")
plt.show()

pad_tok_idx = 77
preds = predicted_val_labels
labs = true_val_labels

new_labels = []
new_preds = []

for pred, lab in zip(preds, labs):
  idx = np.where(lab == pad_tok_idx)[0]
  if len(idx) == 0:
    idx = len(pred)
  else:
    idx = idx[0]

  new_preds.append(pred[:idx])
  new_labels.append(lab[:idx])

new_preds = [item for sublist in new_preds for item in sublist]
new_labels = [item for sublist in new_labels for item in sublist]

print("Classification report: \n",classification_report(new_labels, new_preds))

conf_matrix = confusion_matrix(new_labels, new_preds)
class_labels = list(tag_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Validation Confusion Matrix')
plt.show()

#--------------TESTING LOOP-------------
true_test_labels = []
predicted_test_labels = []

model.eval()
with torch.no_grad():
  for i, (sent, label) in enumerate(test_loader):
      
      sent = sent.to(device)
      label = label.to(device)

      outputs = model(sent)
      to_loss_outputs = outputs.permute(0, 2, 1)
      loss = loss_function(to_loss_outputs, label)

      _, predicted = torch.max(outputs.data, 2)
      total_samples += label.size(0)
      overall_accuracy = calculate_accuracy(predicted, label)

      true_test_labels.extend(label.cpu().numpy())
      predicted_test_labels.extend(predicted.cpu().numpy())


pad_tok_idx = 77
preds = predicted_test_labels
labs = true_test_labels

new_labels = []
new_preds = []

for pred, lab in zip(preds, labs):
  idx = np.where(lab == pad_tok_idx)[0]
  # print(idx, type(idx), len(idx))

  if len(idx) == 0:
    idx = len(pred)
  else:
    idx = idx[0]

  new_preds.append(pred[:idx])
  new_labels.append(lab[:idx])

new_preds = [item for sublist in new_preds for item in sublist]
new_labels = [item for sublist in new_labels for item in sublist]

print("Classification report: ",classification_report(new_labels, new_preds))

cf = confusion_matrix(new_labels, new_preds)
class_labels = list(tag_map.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(cf, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()

torch.save(model, "lstm_model.pt")