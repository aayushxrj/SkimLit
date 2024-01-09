example_input = '''
Hepatitis C virus (HCV) and alcoholic liver disease (ALD), either alone or in combination, count for more than two thirds of all liver diseases in the Western world. 
There is no safe level of drinking in HCV-infected patients and the most effective goal for these patients is total abstinence. Baclofen, a GABA(B) receptor agonist, represents a promising pharmacotherapy for alcohol dependence (AD). 
Previously, we performed a randomized clinical trial (RCT), which demonstrated the safety and efficacy of baclofen in patients affected by AD and cirrhosis. 
The goal of this post-hoc analysis was to explore baclofen's effect in a subgroup of alcohol-dependent HCV-infected cirrhotic patients. 
Any patient with HCV infection was selected for this analysis. Among the 84 subjects randomized in the main trial, 24 alcohol-dependent cirrhotic patients had a HCV infection; 12 received baclofen 10mg t.i.d. and 12 received placebo for 12-weeks. 
With respect to the placebo group (3/12, 25.0%), a significantly higher number of patients who achieved and maintained total alcohol abstinence was found in the baclofen group (10/12, 83.3%; p=0.0123). Furthermore, in the baclofen group, compared to placebo, there was a significantly higher increase in albumin values from baseline (p=0.0132) and a trend toward a significant reduction in INR levels from baseline (p=0.0716). 
In conclusion, baclofen was safe and significantly more effective than placebo in promoting alcohol abstinence, and improving some Liver Function Tests (LFTs) (i.e. albumin, INR) in alcohol-dependent HCV-infected cirrhotic patients. Baclofen may represent a clinically relevant alcohol pharmacotherapy for these patients.
'''

import tensorflow as tf 
import numpy as np 
import spacy
from spacy.lang.en import English

def spacy_fn(abstract):
    nlp = English() # setup English sentence parser
    sentencizer = nlp.add_pipe("sentencizer") # create sentence splitting pipeline object
    doc = nlp(abstract[0]["abstract"])
    abstract_lines = [str(sent) for sent in list(doc.sents)] # return detected sentences from doc in string type (not spaCy token type)
    return abstract_lines

def split_chars(text):
    abstract_chars = [split_chars(sentence) for sentence in text]
    return abstract_chars

def make_prediction(model,text):
    classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    abstract_lines = list()
    
    abstract_lines = spacy_fn(text)
    total_lines_in_sample = len(abstract_lines)

    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
    

    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
    
    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    
    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    # Make predictions on sample abstract features
    test_abstract_pred_probs = model.predict(x=(test_abstract_line_numbers_one_hot,
                                                       test_abstract_total_lines_one_hot,
                                                       tf.constant(abstract_lines),
                                                       tf.constant(abstract_chars)))
    
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    
    test_abstract_pred_classes = [classes[i] for i in test_abstract_preds]

    return (test_abstract_pred_classes, abstract_lines)


# import numpy as np
# from spacy.lang.en import English
# import pandas as pd

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import re
# # from Dataset import SkimlitDataset

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import tensorflow as tf
# import numpy as np

# def pad_sequences(sequences, max_seq_len=0):
#     """Pad sequences to max length in sequence."""
#     max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
#     padded_sequences = np.zeros((len(sequences), max_seq_len))
#     for i, sequence in enumerate(sequences):
#         padded_sequences[i][:len(sequence)] = sequence
#     return padded_sequences
# class SkimlitDataset(Dataset):
#     def __init__(self, text_seq, line_num, total_line):
#         self.text_seq = text_seq
#         self.line_num_one_hot = line_num
#         self.total_line_one_hot = total_line

#     def __len__(self):
#         return len(self.text_seq)

#     def __str__(self):
#         return f"<Dataset(N={len(self)})>"

#     def __getitem__(self, index):
#         X = self.text_seq[index]
#         line_num = self.line_num_one_hot[index]
#         total_line = self.total_line_one_hot[index]
#         return [X, len(X), line_num, total_line]
  
#     def collate_fn(self, batch):
#         """Processing on a batch"""
#         # Getting Input
#         batch = np.array(batch)
#         text_seq = batch[:,0]
#         seq_lens = batch[:, 1]
#         line_nums = batch[:, 2]
#         total_lines = batch[:, 3]

#         # padding inputs
#         pad_text_seq = pad_sequences(sequences=text_seq) # max_seq_len=max_length

#         # converting line nums into one-hot encoding
#         line_nums = tf.one_hot(line_nums, depth=20)

#         # converting total lines into one-hot encoding
#         total_lines = tf.one_hot(total_lines, depth=24)

#         # converting inputs to tensors
#         pad_text_seq = torch.LongTensor(pad_text_seq.astype(np.int32))
#         seq_lens = torch.LongTensor(seq_lens.astype(np.int32))
#         line_nums = torch.tensor(line_nums.numpy())
#         total_lines = torch.tensor(total_lines.numpy())
    
#         return pad_text_seq, seq_lens, line_nums, total_lines
# def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
#         dataloader = DataLoader(dataset=self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle, drop_last=drop_last, pin_memory=True)
#         return dataloader
# def download_stopwords():
#     nltk.download("stopwords")
#     STOPWORDS = stopwords.words("english")
#     porter = PorterStemmer()
#     return STOPWORDS, porter

# def preprocess(text, stopwords):
#     """Conditional preprocessing on our text unique to our task."""
#     # Lower
#     text = text.lower()

#     # Remove stopwords
#     pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
#     text = pattern.sub("", text)

#     # Remove words in paranthesis
#     text = re.sub(r"\([^)]*\)", "", text)

#     # Spacing and filters
#     text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
#     text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
#     text = re.sub(" +", " ", text)  # remove multiple spaces
#     text = text.strip()

#     return text


# def make_skimlit_predictions(text, model, tokenizer, label_encoder): # embedding path
#   # getting all lines seprated from abstract
#   abstract_lines = list()
#   abstract_lines = spacy_fn(text)  
    
#   # Get total number of lines
#   total_lines_in_sample = len(abstract_lines)

#   # Go through each line in abstract and create a list of dictionaries containing features for each line
#   sample_lines = []
#   for i, line in enumerate(abstract_lines):
#     sample_dict = {}
#     sample_dict["text"] = str(line)
#     sample_dict["line_number"] = i
#     sample_dict["total_lines"] = total_lines_in_sample - 1
#     sample_lines.append(sample_dict)

#   # converting sample line list into pandas Dataframe
#   df = pd.DataFrame(sample_lines)
  
#   # getting stopword
#   STOPWORDS, porter = download_stopwords()

#   # applying preprocessing function to lines
#   df.text = df.text.apply(lambda x: preprocess(x, STOPWORDS))

#   # converting texts into numberical sequences
#   text_seq = tokenizer.texts_to_sequences(texts=df['text'])

#   # creating Dataset
#   dataset = SkimlitDataset(text_seq=text_seq, line_num=df['line_number'], total_line=df['total_lines'])

#   # creating dataloader
#   dataloader = dataset.create_dataloader(batch_size=2)

#   # Preparing embedings
# #   embedding_matrix = get_embeddings(embeding_path, tokenizer, 300)

#   # creating model
# #   model = SkimlitModel(embedding_dim=300, vocab_size=len(tokenizer), hidden_dim=128, n_layers=3, linear_output=128, num_classes=len(label_encoder), pretrained_embeddings=embedding_matrix)

#   # loading model weight
# #   model.load_state_dict(torch.load('/content/drive/MyDrive/Datasets/SkimLit/skimlit-pytorch-1/skimlit-model-final-1.pt', map_location='cpu'))

#   # setting model into evaluation mode
#   model.eval()

#   # getting predictions 
#   y_pred = make_prediction(model, dataloader)

#   # converting predictions into label class
#   pred = y_pred.argmax(axis=1)
#   pred = label_encoder.decode(pred)

#   return abstract_lines, pred