# INLP-Assignment 2 : Neural POS Tagging

### Files included:

- Dataset files: These files were used while training and testing the models.

  - en_atis-ud-train.conllu
  - en_atis-ud-dev.conllu
  - en_atis-ud-test.conllu

- Model files: These are the best performing models out of the models tried out. These models will be used for the Part of Speech tagging task. These model files can be found [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ishan_kavathekar_research_iiit_ac_in/EmEHe9wjhuhNtJgCov025U0BRqjgRN3xent5HYO8jumilQ?e=0wrjRt)

  - ffnn_model.pt
  - lstm_model.pt

- Vocabulary files: These files are used while inferring from the models.

  - ffnn_model_vocab.pkl
  - lstm_model_vocab.pkl

- Training and graphs files: These files can be used to reproduce the results and change the hyperparameters of the models.

  - ffnn_train_model.py
  - lstm_train_model.py

- pos_tagger.py: This file is used for inferring from the FFNN and RNN models.
- Report: The report contains all the information of the hyperparameters for the tested model with various graphs and classification metrics.
- README.md

### Instructions to run the files

The <code>pos_tagger.py</code> file can be run using the instruction <code>python3 pos_tagger.py <model_type></code>. Here, model type refers to FFNN or RNN. Use <code>python3 pos_tagger.py f</code> to run the FFNN model and<code>python3 pos_tagger.py r</code> to run the RNN model (LSTM implemented in assignment).<br>
Upon executing the above command, the script takes some time to load the GloVe pretrained embeddings. Once the embeddings have been loaded, the script expects the user to enter a sentence. <br>
After entering the sentence, the model outputs the Part of Speech tag for each of the word from the input sentence. <br>
To avoid loading models again and again, the script takes input sentences until the user types <code>exit</code> (case sensitiv).

### Some implementation details
- Pretrained embeddings from GloVe have been used for FFNN model.
- NLTK tokenizer has been used for tokenzing the sentences. 
- All the punctuation marks are discarded from the input sequence.
- All the words from the input sequence have been converted to lower case before processing them further.

### Example usage:

```
-> python3 pos_tagger.py r

[nltk_data] Downloading package punkt to /home/ishan/nltk_data...
[nltk_data]   Package punkt is already up-to-date!

Enter the sentence: Flight to Denver is delayed.
flight	NOUN
to	ADP
denver	PROPN
is	AUX
delayed	ADJ


Enter the sentence: exit
```
