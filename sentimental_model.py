import pandas as pd
import nltk
import string
from joblib import dump
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.svm import SVC 
nltk.download('punkt')
nltk.download('stopwords')

# Reading dataset
data = pd.read_csv("hatedata.csv", encoding='latin')

# Printing first few rows
print("Original Data:")
print(data.head())

# Function for data preprocessing
def preprocess_text(text):
  
    # Tokenization
    tokens=word_tokenize(text)

    # To Lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Stemming
    porter_stemmer=PorterStemmer()
    stemmed_tokens=[porter_stemmer.stem(token) for token in tokens]

    # Removing stop words
    stop_words=set(stopwords.words('english'))
    filtered_tokens=[token for token in stemmed_tokens if token.lower() not in stop_words]

    #Joining tokens
    preprocess_text=' '.join(filtered_tokens)

    return preprocess_text

# Preprocessing the 'comment' column in the dataset
data['clean_text'] = data['comment'].apply(preprocess_text)

# Preprocessed data
print("\nPreprocessed Data:")
print(data.head())

# Training and testing
from sklearn.model_selection import train_test_split
X = data['clean_text'].iloc[:1000]
y = data['label'].iloc[:1000]

# Split dataset into training and testing sets as in ratio of 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Vectorizing data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Using SVM Model for prediction
classifier = SVC(kernel='linear') 
classifier.fit(X_train_vectors, y_train)

# Saving both the classifier and the vectorizer using joblib
dump((classifier, vectorizer), 'model.joblib')

# Prediction on test set
predictions = classifier.predict(X_test_vectors)

# Evaluation of model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, zero_division=1)

#Displaying accuracy and classification report which contains F1 score, recall, precision
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
