# 1 Data Preprocessing
import pandas as pd 

# 1.1 Load dataset
dataset = pd.read_csv('Dataset/test.csv')

# To show the data, uncomment the lines below 
print(f"Dataset Shape: {dataset.shape}\n")
print(f"Values Count: \n{dataset['rate'].value_counts()}\n")
#print(f"Dataset Head: \n{dataset.head()}\n")

# To show the output counts, uncomment the lines below 
#print(f"Dataset Output Counts: \n{dataset.rate.value_counts()}\n")
#print(f"Dataset Output Counts: \n{dataset.review.value_counts()}\n")

# 1.2 Encode output column into binary if not binary
if dataset['rate'].apply(lambda x: isinstance(x, str)).any():
    dataset.rate.replace('negative', 0, inplace=True)
    dataset.rate.replace('positive', 1, inplace=True)

dataset = dataset[(dataset['rate'] == 1) | (dataset['rate'] == 0)]

# To show the new Data, uncomment the line below 
#print(f"Dataset after change: \n{dataset}\n")

# 1.3 Remove special characters and Convert every word to lowercase
def Clean(text):
    txt = ''
    for i in text:
        if i.isalnum():
            txt = txt + i
        else:
            txt = txt + ' '
    return txt.lower()

dataset.review = dataset.review.apply(Clean)

# To show the Data after cleaning, uncomment the line below 
#print(f"Data after Cleaning: \n{dataset}\n")

