import json
import gzip
import os
from collections import defaultdict
from pke.base import LoadFile
from string import punctuation
from pandas import json_normalize

def load_json_data(filepath):
    """Load JSON data from a file and return a DataFrame with fulltext processing."""
    with open(filepath, 'r', encoding="utf8") as file:
        data = json_normalize([json.loads(line) for line in file])
    data['fulltext'] = data['fulltext'].apply(process_fulltext)
    data.rename(columns={"fulltext": "abstract"}, inplace=True)
    return data['abstract']

def process_fulltext(fulltext):
    """Process fulltext to combine title, abstract, and body text, removing newline characters."""
    title, abstract, main_body = fulltext.split('\n')[1::2]  # Assumes every other line is useful content
    return f"{title}. {abstract} {main_body}".replace('\n', ' ')

def compute_document_frequency_from_data(texts, output_file, stoplist=None, n=3):
    """Compute document frequency from loaded data and save to a gzip file."""
    frequencies, nb_documents = defaultdict(int), 0
    for text in texts:
        doc = LoadFile()
        doc.load_document(input=text, language='en', normalization="stemming")
        doc.ngram_selection(n=n)
        doc.candidate_filtering(stoplist=stoplist)
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1
        nb_documents += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        f.write(f"--NB_DOC--\t{nb_documents}\n")
        for ngram, freq in frequencies.items():
            f.write(f"{ngram}\t{freq}\n")

# Define stop words
stoplist = list(punctuation) + ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

# File paths
files = {
    'NUS': '..\\data\\benchmark_data\\NUS.json',
    'ACM': '..\\data\\benchmark_data\\ACM.json'
}

# Compute document frequencies for each dataset
for name, path in files.items():
    abstracts = load_json_data(path)
    compute_document_frequency_from_data(abstracts, f'doc_freq/{name.lower()}_doc_freq.tsv.gz', stoplist=stoplist)
