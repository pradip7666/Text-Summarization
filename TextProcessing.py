
# TEXT PREPROCESSING
import re
from nltk.stem import WordNetLemmatizer

import re
from nltk.stem import WordNetLemmatizer


# Text preprocessing
def remove_stop_words(text):
    STOPWORDS ={'this', 'll', 'hers', 'few', 'didn', 'whom', 'because', 'shan', 'we', 'too', 'isn', 'yourself', 'be', 've', 'itself', 'from', 'doesn', 'ourselves', 'himself', 'been', 'up', 'she', "mustn't", 'them', "won't", 'its', 'just', "needn't", 'couldn', 'some', 'there', "shouldn't", 'yourselves', 'mustn', 't', 'over', 'weren', 'any', 'until', "wasn't", 'than', 'no', "doesn't", 'yours', 'him', 'being', 'our', 'me', 'then', 'doing', 'here', 'mightn', 'ain', 'ours', 'they', 'but', 'won', "you'd", 'are', "it's", 'haven', 'has', 'had', 'these', 'of', 'were', 'more', 'themselves', 'will', 'm', "aren't", 'and', 'about', 're', 'most', "should've", 'after', 'before', 'while', "hasn't", 'can', 'theirs', 'why', "hadn't", 'below', 'other', 'through', 'was', 'own', 'it', 'did', "you're", 'during', 'where', "you'll", 'have', 'i', 'the', 'to', 'wouldn', 'd', 'on', 'very', 'do', 'against', 'is', 'in', 'now', 'am', 'shouldn', 'if', 'how', 'hasn', 'herself', 'their', "shan't", 'off', "isn't", 'o', 'all', "weren't", 'at', 'further', 'he', "you've", 'between', 'such', 'myself', 'each', 'when', 'once', 'same', 'you', 'by', 'which', 'ma', 'as', "didn't", "don't", 'those', 's', 'or', 'into', 'an', 'so', 'wasn', 'her', "mightn't", 'a', 'y', 'for', 'above', 'aren', "she's", 'what', 'his', 'under', "haven't", "couldn't", 'with', 'needn', 'not', 'again', 'who', 'does', 'my', 'that', 'down', "wouldn't", 'should', 'nor', 'only', 'don', "that'll", 'out', 'your', 'hadn', 'having', 'both'}
    next_text=[]
    for w in text.split():
      if w in STOPWORDS:
        continue
      else:
        next_text.append(w)

    return " ".join(next_text[:])


def preprocess(text):
      text = text.lower() # lowercase all text

      pattern = re.compile('<.*?>') # remove html tags
      text = pattern.sub("", text)

      # remove emoji
      emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags=re.UNICODE)
      text = emoji_pattern.sub(r'', text)

      clear = re.compile(r'https?://\S+|www\.\S+') # remove urls
      text = clear.sub(" ",text)


      text = remove_stop_words(text) # remove stop words

       # Remove references
      text = re.sub(r'\[\d+\]', '', text)  # Removing [21].
      text = re.sub(r'\[\w\]', '', text)   # Removing [j].

      #remove punctuations
      punctuations = '''!"#$%&\'()*+,-/:;?@[\\]^_{|}~`'''
      for char in punctuations:
          text = text.replace(char,"")
      # text = re.sub(r'[^a-z\s.]', '', text)

      # lemmatization
      lemmatizer = WordNetLemmatizer()
      result = []
      for word in text.split():
          result.append(lemmatizer.lemmatize(word))
      text =  " ".join(result)

      # remove numbers
      # text = ''.join([i for i in text if not i.isdigit()])

      return text



def sent_tokenization(text):
      senctences = text.split(".")[:-1]
      senctences = [i.strip() for i in senctences]
      return senctences

def word_tokenization(sentence):
      words = sentence.split(" ")
      return words