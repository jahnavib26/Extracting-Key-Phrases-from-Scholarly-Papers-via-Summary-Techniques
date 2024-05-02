import json
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import pke
import re

# Constants and settings
FILE_PATHS = {
    'ACM': ('..\\data\\benchmark_data\\ACM.json', 'doc_freq/acm_doc_freq.tsv.gz'),
    'NUS': ('..\\data\\benchmark_data\\NUS.json', 'doc_freq/nus_doc_freq.tsv.gz'),
    'SemEval': ('..\\data\\benchmark_data\\semeval_2010.json', 'doc_freq/semeval_2010_doc_freq.tsv.gz')
}

USE_FULLTEXT = 2
MAX_LEN = 400
CURRENT_FILE = 'ACM'

# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        return pd.json_normalize([json.loads(line) for line in file])

# Text preprocessing and paragraph splitting
def preprocess_and_split(data, max_len=400, use_fulltext=2):
    data['abstract'] = data['fulltext'].apply(lambda x: " ".join(x.split("--")[1::2]).replace('\n', ' '))
    if use_fulltext == 2:
        data['abstract'] = data['abstract'].apply(lambda text: split_paragraphs(sent_tokenize(text), max_len))
        data = data.explode('abstract')
    return data

def split_paragraphs(sentences, max_len):
    paragraphs, current_paragraph, current_length = [], '', 0
    for sentence in sentences:
        if current_length + len(word_tokenize(sentence)) <= max_len:
            current_paragraph += ' ' + sentence
            current_length += len(word_tokenize(sentence))
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph, current_length = sentence, len(word_tokenize(sentence))
    paragraphs.append(current_paragraph.strip())
    return paragraphs[:3]

# Extract keyphrases
def extract_keyphrases(data, df_file):
    stemmer = SnowballStemmer('english')
    stopwords_list = set(stopwords.words('english')) | set(string.punctuation)
    stopwords_list.update(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])
    pred_keyphrases, gold_keyphrases = [], []

    for index, text in data.iterrows():
        gold_keyphrases.append([[stemmer.stem(word) for word in phrase.split()] for phrase in data.loc[index, 'keyword'].split(';')])
        extractor = pke.unsupervised.TfIdf()
        extractor.load_document(input=text['abstract'], language='en', normalization="stemming")
        extractor.candidate_selection(n=3, stoplist=stopwords_list)
        extractor.candidate_weighting(df=pke.load_document_frequency_file(input_file=df_file))
        pred_keyphrases.append([kp[0].split() for kp in extractor.get_n_best(n=10)])
    return pred_keyphrases, gold_keyphrases

# Load, process, and extract keyphrases
file_path, df_file = FILE_PATHS[CURRENT_FILE]
data = load_data(file_path)
data = preprocess_and_split(data, MAX_LEN, USE_FULLTEXT)
pred_keyphrases, gold_keyphrases = extract_keyphrases(data, df_file)

# Evaluation
if USE_FULLTEXT == 2:
    traditional_evaluation.evaluation(y_pred=pred_keyphrases, y_test=gold_keyphrases, x_test=data, x_filename='PARAGRAPH', paragraph_assemble_docs=data.index)
else:
    traditional_evaluation.evaluation(y_pred=pred_keyphrases, y_test=gold_keyphrases, x_test=data, x_filename=file_path)
