import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import pycountry
import re
from data_processing import preprocess_text
# Create a TfidfVectorizer object


def perform_query_search(query, tfidf_matrix,documents,keys,vectorizer):
    predicted_documents = []
    
    # Apply the defined functions on the input query
    query = preprocess_text(query)

    #tfidf_matrix = vectorizer.fit_transform(documents)
    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Compute the cosine similarity between the query vector and the documents' TF-IDF vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Get the index of the most similar document
    most_similar_document_index = cosine_similarities.argmax()

    # Get the similarity score of the most similar document
    similarity_score = cosine_similarities[0, most_similar_document_index]

    # Get the indices of the top 10 most similar documents
    top_indices = cosine_similarities.argsort()[0][-20:][::-1]

    
    
    # Get the predicted documents
    predicted_documents += [list(keys)[idx] for idx in top_indices]
    
   

    result={}
    result['predicted_documents']=predicted_documents
    result['count_result']=len(documents)
    
   

    return result  
