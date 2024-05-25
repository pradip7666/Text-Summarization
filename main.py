import TextProcessing as TP
import TF_idf as TF_idf 
from sklearn.cluster import KMeans

def cluster_sent(cluster_labels,sentences):
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
      
        sent = sentences[sentence_id] # fetching orignal sentence from sentences
        clustered_sentences[cluster_id].append(sent) # appending sentence with its position in corpus
    return clustered_sentences


def order_cluster_sent(clustered_sentences,sentence_organizer):
      sent_with_index  = []
      for i in clustered_sentences.values():
            sent_with_index.append([i[0],sentence_organizer[i[0]]]) # add first sentence of each cluster

      # print("sentences with index :\n ", sent_with_index)
      return sorted(sent_with_index, key = lambda x: x[1])

def summarizer(txt,n):
    # split text into sentences
    senctences = txt.split(".")[:-1]
    senctences = [i.strip() for i in senctences]

    # arrange the sentences with there index
    sentence_with_idx = {k:v for v,k in enumerate(senctences)}

    # apply text prepocessing on text to get best result
    corpus = TP.text_processing(txt)

    # split the processed corpus into documents 
    documents = corpus.split(".")[:-1]
    documents = [i.strip() for i in documents]

    # claculate Tf-Idf of each word and convert sentence into vector
    Tf_idf_obj = TF_idf.Tfidfvectorizer()

    Tf_idf_matrix = Tf_idf_obj.fit_transform(documents)
    # print(Tf_idf_matrix)
    # vocab = Tf_idf_obj.vocab

    # cluster the documents on there more similarity
    kmeans = KMeans(n_clusters=n,init='k-means++',max_iter=500,random_state=0)
    kmeans.fit_predict(Tf_idf_matrix)
    cluster_assignment = kmeans.labels_ # cluster Number


    clustered_sentences = cluster_sent(cluster_assignment,senctences)

    sent_with_index = order_cluster_sent(clustered_sentences,sentence_with_idx)

    summary = ""
    for element in sent_with_index:
        summary = summary + element[0].strip() + ". " 

    return summary

