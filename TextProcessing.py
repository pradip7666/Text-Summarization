# Pre-requsites
# !pip install re
# !pip install nltk


# TEXT PREPROCESSING
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def convert_to_lowerCase(text):
      return text.lower()


def remove_html_tags(text):
    pattern = re.compile('<.*?>')

    return pattern.sub("", text)


def remove_emoji(text):
   emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
   return emoji_pattern.sub(r'', text)


def remove_url(text):
   clear = re.compile(r'https?://\S+|www\.\S+')

   return  clear.sub(" ",text)


def remove_stop_words(text):
    STOPWORDS = set(stopwords.words('english'))
    next_text=[]
    for w in text.split():
      if w in STOPWORDS:
        continue
      else:
        next_text.append(w)

    x = next_text[:]
    next_text.clear()
    return " ".join(x)


def remove_punctuations(text):
  puncutuatons =  '''!"#$%&\'()*+,-/:;?@[\\]^_{|}~`''' 

  for char in puncutuatons:    
      text = text.replace(char,"")

  return text


def stemming(text):
    ps = PorterStemmer()
    result = []
    for word in text.split():
      result.append(ps.stem(word))
    return " ".join(result)
 


 
def text_processing(text): 
      text= convert_to_lowerCase(text)
      text = remove_html_tags(text)
      text = remove_emoji(text)
      text =  remove_url(text)
      text = remove_stop_words(text)
      text = remove_punctuations(text)
      text = stemming(text)
      return text

def sent_tokenization(text):
      # return text.split('. ' or '.')[:-1]
      senctences = text.split(".")[:-1]
      senctences = [i.strip() for i in senctences]
      return senctences

def word_tokenization(sentence):
      words = sentence.split(" ")
      words.remove(" ")
      return words
  
