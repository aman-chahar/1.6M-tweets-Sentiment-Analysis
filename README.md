# 1.6M-tweets-Sentiment-Analysis
**Project 2: Sentiment Analysis on 1.6 Million Tweets**

I recently completed a project on sentiment analysis using a dataset of 1.6 million tweets. Here's a breakdown of the code and the project:

**Step 1: Setting Up Kaggle Credentials**
```python
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
```

This code sets up Kaggle credentials for accessing datasets.

**Step 2: Importing Twitter Sentiment Dataset**
```python
!kaggle datasets download -d kazanova/sentiment140
```

This code downloads the Sentiment140 dataset using the Kaggle API.

**Step 3: Extracting Compressed Dataset**
```python
from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
```

The dataset is extracted from the downloaded zip file.

**Step 4: Importing Dependencies**
```python
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
```

This code imports necessary libraries and downloads NLTK stopwords.

**Step 5: Data Processing**
```python
data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1' )

columns_name = ['target', 'id', 'date','flag','user','text']
data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names=columns_name, encoding='ISO-8859-1' )

data.replace({'target':{4:1}}, inplace=True)
```

The data is loaded, columns are named, missing values are checked, and the target column is converted from 4 to 1.

**Step 6: Stemming**
```python
port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ', content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

data['stemmed_content'] = data['text'].apply(stemming)
```

The code defines a stemming function and applies it to the 'text' column, creating a new 'stemmed_content' column.

**Step 7: Data Splitting and Conversion**
```python
x = data['stemmed_content'].values
y = data['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
```

The data is split into training and test sets, and textual data is converted to numerical data using TF-IDF vectorization.

**Step 8: Training the Machine Learning Model**
```python
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
```

A logistic regression model is trained using the training data.

**Step 9: Model Evaluation**
```python
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(y_test, x_test_prediction)
```

The accuracy of the model is evaluated on both training and test data.

**Step 10: Saving the Trained Model**
```python
import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

The trained model is saved using the Pickle library.

**Step 11: Using the Saved Model**
```python
loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))

x_new = x_test[521]
prediction = model.predict(x_new)
print(prediction)
```

The saved model is loaded and used for predictions on new data.

#Python #DataScience #SentimentAnalysis #MachineLearning #NLP #Kaggle```
