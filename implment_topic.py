import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from data_processing import preprocess_text
vectorizer = TfidfVectorizer()
# Query processing
def implmentTopicAlg(query,tfidf_matrix, temp, data_title,topic_weights,keys):
    pureQuery = query

    # Apply the defined functions on the input query
    query=preprocess_text(query)

    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Cosine Similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_document_index = cosine_similarities.argmax()
    similarity_score = cosine_similarities[0, most_similar_document_index]

    # Topic Algorithm
    most_similar_topic_weights = topic_weights[most_similar_document_index]
    best_topic_index = most_similar_topic_weights.argmax()
    documents_in_best_topic = [i for i, weight in enumerate(topic_weights[:, best_topic_index]) if weight > 0]
    topic_similarities = cosine_similarities[0, documents_in_best_topic]
    sorted_indices_within_topic = np.array(documents_in_best_topic)[np.argsort(topic_similarities)[::-1]]

    top_indices_within_topic = sorted_indices_within_topic[:10]
    print("The result of Topic Algorithm search : ",len(top_indices_within_topic))
    predicted_documents += [keys[idx] for idx in sorted_indices_within_topic]
    # Print the most similar documents within the best topic and their similarity scores
    for idx in top_indices_within_topic:
        most_similar_document_key = list(temp.keys())[idx]
        print("Title:", data_title[str(most_similar_document_key)])

    # Print the best-weighted topic and its weight
    print("Best-weighted topic:")
    print("Topic", best_topic_index, "- Weight:", most_similar_topic_weights[best_topic_index])
