import re
import json
import pandas as pd
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pke

# Constants for file paths and settings
FILE_PATHS = {
    'acm': '..\\data\\benchmark_data\\ACM.json',
    'nus': '..\\data\\benchmark_data\\NUS.json',
    'semeval': '..\\data\\benchmark_data\\semeval_2010.json',
    'acm_sum': '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv',
    'nus_sum': '..\\data\\benchmark_data\\summarization_experiment\\NUS_summarized.csv',
    'semeval_sum': '..\\data\\benchmark_data\\summarization_experiment\\SemEval-2010_summarized.csv'
}

# Configuration based on the choice of dataset
file = FILE_PATHS['nus']  # Modify as per dataset requirement
use_fulltext = 0
MAX_LENGTH = 250  # Original was 220, modified to 250

# Functions to read and process data
def load_data(file_path):
    if 'json' in file_path:
        with open(file_path, 'r', encoding="utf8") as file:
            data = pd.json_normalize([json.loads(line) for line in file])
    else:
        data = pd.read_csv(file_path, encoding="utf8")
    return data

# Processing text data
def process_text(data, use_fulltext):
    text_field = 'abstract' if 'abstract' in data.columns else 'fulltext'
    data[text_field] = data[text_field].apply(lambda x: re.sub(r'(?<!\w)([A-Z])\.', r'\1', x.replace('e.g.', 'eg').replace('i.e.', 'ie').replace('etc.', 'etc')))
    data[text_field] = data[text_field].apply(lambda text: ' '.join(sent_tokenize(text)).replace('\n', ' '))

    if use_fulltext == 2:
        data['abstract'] = data[text_field].apply(split_into_paragraphs, max_len=MAX_LENGTH)
        data = data.explode('abstract')
    elif use_fulltext == 1:
        data['abstract'] = data.apply(lambda row: f"{row['title']}. {row[text_field]}", axis=1)
    else:
        data['abstract'] = data.apply(lambda row: f"{row['title']}. {row['abstract']}", axis=1)

    return data

# Split document into paragraphs
def split_into_paragraphs(doc, max_len=250):
    paragraphs, paragraph, current_length = [], '', 0
    for sentence in sent_tokenize(doc):
        if current_length + len(word_tokenize(sentence)) <= max_len:
            paragraph += ' ' + sentence
            current_length += len(word_tokenize(sentence))
        else:
            paragraphs.append(paragraph.strip())
            paragraph, current_length = sentence, len(word_tokenize(sentence))
    if paragraph:
        paragraphs.append(paragraph.strip())
    return paragraphs[:3]  # Limit to first 3 paragraphs

# Extract keyphrases using MultipartiteRank
def extract_keyphrases(data):
    stemmer = SnowballStemmer('english')
    stopwords_list = set(stopwords.words('english')) | set(string.punctuation)
    stopwords_list.update(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])
    gold_keyphrases, pred_keyphrases = [], []

    for abstract in data['abstract']:
        gold_keyphrases.append([[stemmer.stem(word) for word in keyphrase.split()] for keyphrase in abstract.get('keyword', '').split(';')])
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=abstract, normalization="stemming")
        extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, stoplist=stopwords_list)
        extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
        pred_keyphrases.append([kp[0].split() for kp in extractor.get_n_best(n=10)])

    return pred_keyphrases, gold_keyphrases

# Main processing sequence
data = load_data(file)
data = process_text(data, use_fulltext)
pred_keyphrases, gold_keyphrases = extract_keyphrases(data)

# Output results
print(pred_keyphrases)
print(gold_keyphrases)


