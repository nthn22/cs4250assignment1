#-------------------------------------------------------------------------
# AUTHOR: Nathan Zamora
# FILENAME: assignment1indexing
# SPECIFICATION: Output tf-idf document matrix following requirements from question 7.
# FOR: CS 4250- Assignment #1
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#Importing some Python libraries
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd



documents = []

#Reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

#Conducting stopword removal for pronouns/conjunctions. Hint: use a set to define your stopwords.
#--> add your Python code here

stopWords = {'I', 'and', 'She', 'her', 'They', 'their'}

newdocument = []
for doc in documents:
    words = doc.split()
    filtered_words = [word for word in words if word not in stopWords]
    newdocument.append(' '.join(filtered_words))

print(newdocument)


# Conducting stemming (mapping word variations to their stems)
stemming = {"loves": "love", "love": "love", "cats": "cat", "cat": "cat", "dogs": "dog", "dog": "dog"}

# Identifying the index terms
terms = ["Love", "Cat", "Dog"]
# Convert terms to lowercase for consistency
terms = [term.lower() for term in terms]

# Getting word counts (initialize dictionary for counting terms)

term_counts_per_document = []

# Loop through each document in newdocument
for doc in newdocument:
    # Split each document into words and convert to lowercase
    words = doc.lower().split()
    word_counts = {term: 0 for term in terms}
    # Loop through each word
    for word in words:
        # Check if word is in terms or its stemmed version exists in stemming
        stemmed_word = stemming.get(word, word)  # Get stemmed version or keep the word itself
        if stemmed_word in word_counts:
            word_counts[stemmed_word] += 1
    term_counts_per_document.append(word_counts)        



for idx, counts in enumerate(term_counts_per_document):
    print(f"Document {idx + 1}: {counts}")



# Using TfidfVectorizer to build the document-term matrix
vectorizer = TfidfVectorizer(vocabulary=terms)  # Specify the vocabulary (terms to include)

# Transform the newdocument into the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(newdocument)
    
#Building the document-term matrix by using the tf-idf weights.
#--> add your Python code here

docTermMatrix = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)   

#Printing the document-term matrix.
print("Document-Term Matrix (TF-IDF Weights):")
print(docTermMatrix)