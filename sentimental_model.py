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

# Load dataset from CSV file
data = pd.read_csv("hatedata.csv", encoding='latin')

# Check the first few rows of the dataset
print("Original Data:")
print(data.head())


# PREPROCESSING THE DATA 
def preprocess_text(text):
  
    # TOKENIZATION
    tokens=word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # STEMMING
    porter_stemmer=PorterStemmer()
    stemmed_tokens=[porter_stemmer.stem(token) for token in tokens]

    # REMOVING STOP WORDS
    stop_words=set(stopwords.words('english'))
    filtered_tokens=[token for token in stemmed_tokens if token.lower() not in stop_words]

    #JOINING THE TOKENS
    preprocess_text=' '.join(filtered_tokens)

    return preprocess_text


# Preprocess the 'text' column in the dataset
data['clean_text'] = data['comment'].apply(preprocess_text)

# Display the preprocessed data
print("\nPreprocessed Data:")
print(data.head())


from sklearn.model_selection import train_test_split

X = data['clean_text'].iloc[:1000]
y = data['label'].iloc[:1000]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Vectorize text data using bag-of-words representation
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


classifier = SVC(kernel='linear')  # Use a linear kernel for SVM
classifier.fit(X_train_vectors, y_train)


# Save both the classifier and the vectorizer
dump((classifier, vectorizer), 'model.joblib')


# Predict on test set
predictions = classifier.predict(X_test_vectors)


# Evaluate model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, zero_division=1)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)