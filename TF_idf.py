import numpy as np

class Tfidfvectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, documents):
        # Calculate TF
        tf_matrix = self.calculate_tf(documents)

        # Calculate IDF
        self.calculate_idf(documents)

        # Calculate TF-IDF
        tfidf_matrix = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            for j, term in enumerate(self.vocab):
                if term in tf_matrix[i]:
                    tfidf_matrix[i, j] = tf_matrix[i][term] * self.idf[term]

        return tfidf_matrix

    def calculate_tf(self, documents):
        tf_matrix = []
        for doc in documents:
            tf_doc = {}
            words = doc.split()
            word_count = len(words)
            for word in words:
                tf_doc[word] = tf_doc.get(word, 0) + 1 / word_count
            tf_matrix.append(tf_doc)
            
        return tf_matrix

    def calculate_idf(self, documents):
        doc_count = len(documents)
        for doc in documents:
            words = set(doc.split())
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0) + 1

        for term, freq in self.vocab.items():
            self.idf[term] = np.log(doc_count / (freq))  


















