import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In case a downlaoded is required uncomment the line below, afterwards, you can remove it
#import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

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

# 2 Data Lemmatization
def remove_stopword(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

dataset.review = dataset.review.apply(remove_stopword)

def stem_txt(text):
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in text])

dataset.review = dataset.review.apply(stem_txt)

#def lemmatize_text(text):
#    lemmatizer = WordNetLemmatizer()
#    return " ".join([lemmatizer.lemmatize(w, pos=wordnet.VERB) for w in text])
#
#dataset.review = dataset.review.apply(lemmatize_text)
######################################################################################################################
# 3 Model Creation
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 3.1 Creating Bag Of Words (BOW)
reviews = np.array(dataset.review.values)
sentiment = np.array(dataset.rate.values) 
countvector = CountVectorizer(max_features = 2000)
reviews_bow = countvector.fit_transform(dataset.review).toarray()

# 3.3 Split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(reviews_bow, sentiment, test_size=0.2, random_state=13)

# To see the count, uncomment the line below 
#print(f"Train shapes : X = {X_train.shape}, y = {y_train.shape}")
#print(f"Test shapes  : X = {X_test.shape},  y = {y_test.shape}\n")

## 3.4 Defining the RNN Model
model = Sequential([
    Embedding(input_dim = reviews_bow.shape[1], output_dim = 128, input_length = X_train.shape[0]),
    LSTM(64,return_sequences = True),
    Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train_reshaped = y_train.reshape(-1, 1)
model.fit(X_train, y_train_reshaped, epochs=10, batch_size=32)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")