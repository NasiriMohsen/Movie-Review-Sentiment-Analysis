import pandas as pd 

# 1 Data Preprocessing
dataset = pd.read_csv('Dataset/test.csv')

if dataset['rate'].apply(lambda x: isinstance(x, str)).any():
    dataset.rate.replace('negative', 0, inplace=True)
    dataset.rate.replace('positive', 1, inplace=True)

dataset = dataset[(dataset['rate'] == 1) | (dataset['rate'] == 0)]

def Clean(text):
    txt = ''
    for i in text:
        if i.isalnum():
            txt = txt + i
        else:
            txt = txt + ' '
    return txt.lower()

dataset.review = dataset.review.apply(Clean)
######################################################################################################################
# 2 Data Lemmatizing or Stemming (I prefer Lemmatizing, it just feels right)
# In case of an error or a downlaod require uncomment the line below after downlaod you can remove it
#import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer # The SnowballStemmer is based on the Porter algorithim but improved

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# 2.1 Removing Stopwords
def remove_stopword(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

dataset.review = dataset.review.apply(remove_stopword)

# To show the the resault, uncomment the line below 
#print(f"resault of removing the Stopwords: \n{dataset.review}\n")

## 2.2.1 Stemming
#def stem_txt(text):
#    stemmer = SnowballStemmer('english')
#    return " ".join([stemmer.stem(w) for w in text])
#
#dataset.review = dataset.review.apply(stem_txt)
#
## To show the the resault, uncomment the line below 
##print(f"After stemming: \n{dataset.review}\n")

# 2.2.2 lemmatizing
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w, pos=wordnet.VERB) for w in text])

dataset.review = dataset.review.apply(lemmatize_text)

# To show the the resault, uncomment the line below 
#print(f"Review sample after lemmatizing the words:\n{dataset.review}\n")