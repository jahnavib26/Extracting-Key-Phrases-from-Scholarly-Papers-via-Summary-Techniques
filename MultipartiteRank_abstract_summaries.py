import json
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pke

# Define file paths
FILE_ABSTRACT = '..\\data\\benchmark_data\\ACM.json'
FILE_SUMMARIES = '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv'

# Read and process JSON data
def load_json_data(filepath):
    with open(filepath, 'r', encoding="utf8") as file:
        return pd.json_normalize([json.loads(line) for line in file])

# Read CSV data
def load_csv_data(filepath):
    return pd.read_csv(filepath, encoding="utf8")

# Combine title and abstract, clean newlines
def process_abstracts(data, source_type):
    if source_type == 'json':
        data['abstract'] = data['fulltext'].apply(lambda x: x.split('--A\n')[1].split('--B\n')[0].replace('\n', ' '))
        data['title'] = data['fulltext'].apply(lambda x: x.split('--T\n')[1].split('--A\n')[0])
        data['fulltext'] = data['title'] + ' ' + data['abstract']
        data.rename(columns={"fulltext": "abstract"}, inplace=True)
    else:
        data['abstract'] = data.apply(lambda row: f"{row['title']}. {row['abstract'].replace('\n', ' ')}", axis=1)
    return data

# Extract keyphrases using MultipartiteRank
def extract_keyphrases(data):
    stemmer = SnowballStemmer('english')
    stopwords_list = set(stopwords.words('english')) | set(string.punctuation)
    stopwords_list.update(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])
    pred_keyphrases, gold_keyphrases = [], []

    for index, row in data.iterrows():
        # Gold keyphrases
        gold_keyphrases.append([stemmer.stem(word) for word in row['keyword'].split(';')])

        # Predict keyphrases
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=row['abstract'], normalization='stemming')
        extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, stoplist=stopwords_list)
        extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
        pred_keyphrases.append([kp[0].split() for kp in extractor.get_n_best(n=10)])

    return pred_keyphrases, gold_keyphrases

# Load and process data
data_abstract = load_json_data(FILE_ABSTRACT)
data_summaries = load_csv_data(FILE_SUMMARIES)
data_abstract = process_abstracts(data_abstract, 'json')
data_summaries = process_abstracts(data_summaries, 'csv')

# Extract keyphrases
pred_keyphrases_abstract, gold_keyphrases_abstract = extract_keyphrases(data_abstract)
pred_keyphrases_summaries, _ = extract_keyphrases(data_summaries)

# Combine abstracts and summaries
data_summaries['abstract'] = data_abstract['abstract'] + ' ' + data_summaries['abstract']

# Combine keyphrases
for index, keyphrases in enumerate(pred_keyphrases_abstract):
    keyphrases.extend(pred_keyphrases_summaries[index])

# Evaluate model performance
# Since the traditional_evaluation module's structure is not provided, assumed it has a function called `evaluation`
# traditional_evaluation.evaluation(y_pred=pred_keyphrases_abstract, y_test=gold_keyphrases_abstract, x_test=data_summaries, x_filename='')

# Optionally, display or process results
print(data_summaries['abstract'].head())
print(pred_keyphrases_abstract)
