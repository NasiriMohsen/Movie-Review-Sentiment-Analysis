# Moive Review Sentiment Analyzer  
import numpy as np
import pandas as pd 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Loades the dataset 
def load_dataset(path='Dataset/test.csv'):
    dataset = pd.read_csv(path)

    if dataset['rate'].apply(lambda x: isinstance(x, str)).any():
        dataset.rate.replace('negative', 0, inplace=True)
        dataset.rate.replace('positive', 1, inplace=True)

    dataset = dataset[(dataset['rate'] == 1) | (dataset['rate'] == 0)]
    return dataset

# 1 Cleans the text from special characters 
def Clean(text):
    txt = ''
    for i in text:
        if i.isalnum():
            txt = txt + i
        else:
            txt = txt + ' '
    return txt.lower()

# 2 Removes stopwords from text 
def remove_stopword(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

# 3 Lemmatizes text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w, pos=wordnet.VERB) for w in text])

# Model Creation:
dataset = load_dataset()
dataset.review = dataset.review.apply(Clean) #1 Clean 
dataset.review = dataset.review.apply(remove_stopword) #2 Remove Stopwords
dataset.review = dataset.review.apply(lemmatize_text) #3 lemmatize_text

max_len = 100
sentiment = np.array(dataset.rate.values) # commonly known as y_train
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(dataset['review'])
sequences = tokenizer.texts_to_sequences(dataset['review'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post') # commonly known as X_train

# Defining the RNN layers
model = Sequential([
    Embedding(input_dim=tokenizer.num_words, output_dim=128, input_length=max_len),
    LSTM(64,return_sequences = True),
    Lambda(lambda x: x[:, -1, :]),
    Dense(units=1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(padded_sequences, sentiment, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(padded_sequences, sentiment)
print(f"Test Loss: {round(loss,4)}, Test Accuracy: {round(accuracy)}")

# Run a loop for predicitons 
while True:
    # asking for review and proccesing it  
    new_review = lemmatize_text(remove_stopword(Clean(input("\x1b[36mEnter the movie review to recive the sentiment of it: \x1b[0m\n"))))

    new_sequence = tokenizer.texts_to_sequences([new_review])
    new_sequence = pad_sequences(new_sequence, maxlen=max_len, padding='post')
    
    # evaluating the reviews sentiment
    prediction = model.predict(new_sequence)[0][0]
    print(f"\x1b[36mThe prediction score is\x1b[33m {round(prediction*100, 2)}\x1b[36m and the sentiment of the movie review is:\x1b[0m")

    if prediction > 0.5:
      print("\x1b[32mPositive \x1b[0m")
    else:
      print("\x1b[31mNegative \x1b[0m")




