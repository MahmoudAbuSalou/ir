**Project Overview:**
we used to data set :
1- [beir] (https://ir-datasets.com/beir.html#beir/trec-covid)
2- [covid19] (https://ir-datasets.com/cord19.html#cord19/trec-covid/round4)
my project focuses on creating a search engine for two datasets: COVID-19 and BIRE.
The project includes several preprocessing steps, the creation of a TF-IDF matrix, an inverted index, and various enhancements.
**Preprocessing Steps:**
List and describe each preprocessing step :
1. replace_percent_text for replace % with percent
2. expand_country_names usinr packeg pycountry
3. clean_text for Remove punctuations
4. lower_text for Convert to lowercase
5. lemmatize_text using WordNetLemmatizer
6. stem_text using PorterStemmer
7. remove_stopwords using stopwords
8. correct_spelling using TextBlob
**TF-IDF Matrix and Inverted Index:**
calculate tfidf for documints via TfidfVectorizer 
**Query Processing and Representation:**
query is processed and represented using the same steps as in the preprocessing section.
query is tokenized, cleaned, stemmed, and transformed to match the format of the TF-IDF matrix.
**Cosine Similarity Calculation:**
The cosine similarity is a measure that quantifies the similarity between two vectors in a high-dimensional space. In the context of a search engine, it is commonly used to calculate the similarity between a query and documents in the dataset.
**Enhancements:**
enhancement I mentioned (topic algorithm using LatentDirichletAllocation, cluster algorithm using KMeans, query suggestions).
 how each enhancement improves the search engine's performance or user experience:
 Rank the documents based on their cosine similarity values. Higher cosine similarity values indicate higher similarity to the query.
 suggistions:
 
 
 ![photo_2023-06-14_00-07-13](https://github.com/MahmoudAbuSalou/ir/assets/87876527/1b6f57e4-3737-405f-9e32-2533179da812)
 
 
 using clustring Reducing the search circle within the documants:
 
 
 ![Screenshot (153)](https://github.com/MahmoudAbuSalou/ir/assets/87876527/3cbd1eef-4187-4278-826f-d226f38e81f6)
 
 
 
**Evaluation Metrics:**
 the evaluation metrics you calculated (Precision@10, Precision, Recall, MAP, MRR).
 
 
 ![Screenshot (152)](https://github.com/MahmoudAbuSalou/ir/assets/87876527/d398786e-07b6-4b76-8d3d-eef09a541481)
 
 
 **mpoile app to active and interaction on data sets**
 
 
 ![photo_2023-06-14_00-06-49](https://github.com/MahmoudAbuSalou/ir/assets/87876527/d9dfb0d1-af1d-40ae-a247-ea8445a902a4)
 
 

 **Conclusion:**
1.Information Retrieval: The search engine provides a means for users to retrieve relevant documents from the COVID-19 and BIRE datasets based on their queries.     It leverages various techniques to improve retrieval accuracy and efficiency.
2. TF-IDF Matrix and Inverted Index: The project calculates the TF-IDF matrix for the documents in the datasets. This matrix represents the importance of each        term in each document, enabling efficient retrieval based on term relevance. Additionally, the creation of an inverted index allows for quick document     rtrieval based on terms, significantly speeding up the search process.
 
 Overall, the search engine project aims to provide a robust, efficient, and user-friendly solution for information retrieval from the COVID-19 and BIRE datasets. It combines various techniques, algorithms, and evaluation measures to deliver relevant search results and enhance the overall search experience.
 
