import pandas as pd
import json
from scipy.sparse import load_npz
from flask import Flask, request, jsonify
from implment_cluster import readFileCluster, implmentClusterAlg
from query_processing import perform_query_search
from implment_topic import implmentTopicAlg
from eval import eval
# Create an instance of Flask
app = Flask(__name__)
data_set=1
# Provide the path to your JSONL file


jsonl_file_path = r'C:\Users\User\.ir_datasets\beir\webis-touche2020\webis-touche2020\corpus.jsonl'
if(data_set==2):
    jsonl_file_path = r'C:\Users\User\.ir_datasets\cord19\2020-06-19\files\corpus.jsonl'
# Read the JSONL file into a DataFrame
dff = pd.read_json(jsonl_file_path, lines=True)

# Create dictionaries for data_title and data_text
data_title = {}
data_text = {}
k=0
for i, row in dff.iterrows():
    if(k<10):
        print(row['_id'])
        k+=1
    data_title[str(row['_id'])] = str(row['title'])
    data_text[str(row['_id'])] = str(row['text'])

# Load the TF-IDF matrix from the file

file_path = 'C:/Users/User/.ir_datasets/beir/webis-touche2020/webis-touche2020/tfidf_matrix.npz'
if(data_set==2):
    file_path = 'C:/Users/User/.ir_datasets/cord19/2020-06-19/files/tfidf_matrix.npz'

tfidf_matrix = load_npz(file_path)

# Load the corpus JSON
corpus_json_path = 'C:/Users/User/.ir_datasets/beir/webis-touche2020/webis-touche2020/corpus.jsonl'
if(data_set==2):
    file_path = 'C:/Users/User/.ir_datasets/cord19/2020-06-19/files/corpus.jsonl'
temp =  pd.read_json(corpus_json_path, lines=True)

# Read cluster labels
cluster_labels = readFileCluster()

with open(r'C:\Users\User\.ir_datasets\beir\webis-touche2020\webis-touche2020\documents.json','r') as f:
    documents = json.load(f)
if(data_set==2):
    with open(r'C:/Users/User/.ir_datasets/cord19/2020-06-19/files/documents.json','r') as f:
        documents = json.load(f)
with open(r'C:\Users\User\.ir_datasets\beir\webis-touche2020\webis-touche2020\keys.json','r') as f:
    keys = json.load(f)
if(data_set==2):
    with open(r'C:/Users/User/.ir_datasets/cord19/2020-06-19/files/keys.json','r') as f:
        keys = json.load(f)
@app.route('/get_query', methods=['POST'])
def predict():
    query = request.form['kewword']
     
    predicted_documents = perform_query_search(query, tfidf_matrix,documents, temp, data_title,data_text,keys)
    
    evaluations=eval(query)
    return jsonify(predicted_documents)

@app.route('/get_query_with_cluster', methods=['POST'])
def predictWithCluster():
    query = request.form['kewword']
    predicted_documents = implmentClusterAlg(query, tfidf_matrix, temp, data_title, cluster_labels,keys)
    evaluations=eval(query)
    return jsonify(predicted_documents)

@app.route('/get_query_with_topic', methods=['POST'])
def predictWithTopic():
    query = request.form['kewword']
    predicted_documents = implmentTopicAlg(query, tfidf_matrix, temp, data_title, cluster_labels,keys)
    evaluations=eval(query)
    return jsonify(predicted_documents)

if __name__ == '__main__':
    app.debug = True
    app.run()
