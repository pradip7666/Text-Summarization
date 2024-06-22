import gensim
import numpy as np
from nltk.tokenize import sent_tokenize
import Kmedoid as kmediod
import TextProcessing as TP


path = "\model\word2vec-google-news-300"
model = gensim.models.KeyedVectors.load(path)


#sent to vector with adding word vector and normalize it
def sent_to_vec(sent):
  vector_size = model.vector_size
  vector = np.zeros(vector_size)
  count = 0
  for word in sent:
    if word in model:
      vector = np.add(vector, model.get_vector(word))
      count += 1

  if count != 0:
    vector /= count
  return vector


# cluster the sentences of there clusters
def cluster_sent(cluster_labels,sentences):
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        sent = sentences[sentence_id] # fetching orignal sentence from sentences
        clustered_sentences[cluster_id].append(sent) # appending sentence with its position in corpus
    return clustered_sentences




def summarizer(txt,n):
    # split text into sentences
    senctences = sent_tokenize(txt)
    senctences = [i.strip() for i in senctences]

    # arrange the sentences with there index
    sentence_with_idx = {k:v for v,k in enumerate(senctences)}

    # split the processed corpus into documents
    documents = [TP.preprocess(i) for i in senctences]

    # arrange the sentences with there index
    documents_with_idx = {k:v for v,k in enumerate(documents)}

    # mapping vector with sent
    sent_vectors = {}
    for i in documents:
        vector = sent_to_vec(i)
        sent_vectors[i] = vector

    # typecasting the sent vectors to numpy array
    matrix_vec = sent_vectors.values()
    matrix_vec = np.array(list(matrix_vec))

    # applying the kmean clustering
    # cluster_assignment,cluster_centers = clustering(matrix_vec,n)

    # clustered sentences on there respective clusters
    # clustered_sentences = cluster_sent(cluster_assignment,senctences)


    # ordering the clusters sentences according to there order in text
    # ordered_summary_sent = order_cluster_sent(clustered_sentences,sentence_with_idx)

    n = 10 # number of clusters

    medoids,labels = kmediod.k_medoids(matrix_vec, n, max_iter=300)

    summary= ""
    documents_indexes = []
    for i in matrix_vec:
        if i in medoids:
            for k,v in sent_vectors.items():
                if np.array_equal(v,i):
                    documents_indexes.append(documents_with_idx[k])

    for i in sorted(documents_indexes):
        for k,v in sentence_with_idx.items():
            if v == i:
                summary = summary + k.strip() 
    
    return summary