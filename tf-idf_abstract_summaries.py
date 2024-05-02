import json
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pke

def load_data(json_file, csv_file):
    """ Load data from a JSON file and a CSV file, returning pandas DataFrames. """
    with open(json_file, 'r', encoding="utf8") as file:
        data_abstract = pd.json_normalize([json.loads(line) for line in file])

    data_summaries = pd.read_csv(csv_file, encoding="utf8")
    return data_abstract, data_summaries

def combine_text(data, file_type='json'):
    """ Combine title and abstract, remove newlines, and update the dataframe. """
    if file_type == 'json':
        data['abstract'] = data['fulltext'].apply(lambda x: " ".join(x.split('\n')[1::2])).str.replace('\n', ' ')
        data.rename(columns={"fulltext": "abstract"}, inplace=True)
    else:
        data['abstract'] = data.apply(lambda row: f"{row['title']}. {row['abstract']}", axis=1).str.replace('\n', ' ')
    if 'keywords' in data.columns:
        data.rename(columns={"keywords": "keyword"}, inplace=True)
    return data

def extract_keyphrases(data, df_file):
    """ Extract keyphrases using TfIdf and return predictions and gold keyphrases. """
    stoplist = set(stopwords.words('english')) | set(string.punctuation)
    stoplist.update(['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'])
    gold_keyphrases, pred_keyphrases = [], []

    for index, text in data['abstract'].iteritems():
        gold_keyphrases.append([[SnowballStemmer('english').stem(word) for word in phrase.split()] for phrase in data.loc[index, 'keyword'].split(';')])
        extractor = pke.unsupervised.TfIdf()
        extractor.load_document(input=text, language='en', normalization="stemming")
        extractor.candidate_selection(n=3, stoplist=stoplist)
        extractor.candidate_weighting(df=pke.load_document_frequency_file(input_file=df_file))
        pred_keyphrases.append([kp[0].split() for kp in extractor.get_n_best(n=10)])

    return pred_keyphrases, gold_keyphrases

# File paths
file_abstract = '..\\data\\benchmark_data\\ACM.json'
file_summaries = '..\\data\\benchmark_data\\summarization_experiment\\ACM_summarized.csv'
df_file = 'doc_freq/acm_doc_freq.tsv.gz'

# Load and process data
data_abstract, data_summaries = load_data(file_abstract, file_summaries)
data_abstract = combine_text(data_abstract, 'json')
data_summaries = combine_text(data_summaries, 'csv')

# Extract keyphrases
pred_keyphrases_abstract, gold_keyphrases = extract_keyphrases(data_abstract, df_file)
pred_keyphrases_summaries, _ = extract_keyphrases(data_summaries, df_file)

# Combine abstracts and summaries
data_summaries['abstract'] = data_abstract['abstract'] + ' ' + data_summaries['abstract']

# Combine keyphrases
for idx, kps in enumerate(pred_keyphrases_abstract):
    kps.extend(pred_keyphrases_summaries[idx])

# Evaluation (assuming traditional_evaluation.evaluation function is defined properly elsewhere)
traditional_evaluation.evaluation(y_pred=pred_keyphrases_abstract, y_test=gold_keyphrases, x_test=data_summaries, x_filename='')

print("Finished processing and evaluation.")
