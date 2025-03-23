import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load the Dataset
def load_data(filepath):
    """
    Loads the dataset and ensures proper column selection.
    """
    try:
        data = pd.read_csv(filepath, encoding='latin-1')  # Adjust encoding if necessary
        
        # Rename columns if dataset contains 'v1' (label) and 'v2' (text)
        if 'v1' in data.columns and 'v2' in data.columns:
            data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

        # Convert labels to numeric (0 = Ham, 1 = Spam)
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        
        print("Data Loaded Successfully!")
        print(data.head())  # Print sample data to verify structure
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# 2. Preprocess the Text
def preprocess_text(text):
    """
    Cleans and preprocesses the SMS text:
    - Removes special characters
    - Converts to lowercase
    - Removes stopwords
    - Performs lemmatization
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I)  # Remove non-alphanumeric characters
    text = text.lower()
    tokens = text.split()
    
    # Load stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

# 3. Feature Extraction
def extract_features(text_data, vectorizer=None):
    """
    Converts text data into TF-IDF feature vectors.
    If a vectorizer is provided, transforms the text using the existing vectorizer.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocab size
        features = vectorizer.fit_transform(text_data)
    else:
        features = vectorizer.transform(text_data)

    return features, vectorizer

# 4. Train a Model
def train_model(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes classifier.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# 5. Evaluate the Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints performance metrics.
    """
    predictions = model.predict(X_test)
    print("\n=== Model Evaluation ===")
    print("Accuracy Score:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

# --- Main Execution ---

# 1. Load dataset
filepath = "C:/Users/rathn/OneDrive/ドキュメント/GrowthLink/dataset_spamSMS/spam.csv"  # Update with actual file path
data = load_data(filepath)

if data is None:
    exit()

# 2. Preprocess text
data['processed_text'] = data['text'].apply(preprocess_text)

# 3. Choose Features and Target Variable
X = data['processed_text']
y = data['label']

# 4. Split into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Extract Features (TF-IDF)
X_train_features, vectorizer = extract_features(X_train.tolist())  # Convert to list for TF-IDF
X_test_features, _ = extract_features(X_test.tolist(), vectorizer)

# 6. Train the Model
model = train_model(X_train_features, y_train)

# 7. Evaluate the Model
evaluate_model(model, X_test_features, y_test)
