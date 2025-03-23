# Spam-SMS-Detection

Overview:
This project is a Spam SMS Detection System that classifies messages as Spam or Ham (legitimate) using Natural Language Processing (NLP) and Machine Learning. The system uses TF-IDF vectorization and a Naïve Bayes classifier to efficiently classify text messages.

Features:
Loads and preprocesses SMS dataset
Extracts features using TF-IDF Vectorization
Trains a Multinomial Naïve Bayes model for classification
Evaluates model performance using accuracy, precision, recall, and F1-score
Predicts whether an SMS is spam or ham

Dataset:
The dataset consists of SMS messages labeled as spam or ham. The dataset file spam.csv has the following structure:

v1,v2
label,text
ham, "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
ham, "Ok lar... Joking wif u oni..."
spam, "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"

v1 → Label (ham/spam)

v2 → SMS text

Note: The dataset contains some unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4), which are ignored during processing.

Dowmload the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Installation & Setup:
1. Clone the Repository

git clone https://github.com/your-username/Spam-SMS-Detection.git
cd Spam-SMS-Detection

2. Install Dependencies
Make sure you have Python installed, then install required libraries:

pip install pandas scikit-learn nltk

3. Download NLTK Resources

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Running the Project:
Place the dataset (spam.csv) in the project folder.

Run the Python script:

python spam_sms.py

Model Training & Evaluation:
Preprocessing: Cleans, tokenizes, and lemmatizes text.
Feature Extraction: TF-IDF vectorization is applied.
Training: The Naïve Bayes classifier is trained on processed SMS data.
Evaluation: Model performance is measured using a test set.
Prediction: The trained model predicts whether SMS messages are spam or ham.

Sample Output:

=== Model Evaluation ===
Accuracy Score: 97%
Confusion Matrix:
[[480  10]
 [  8  52]]
Classification Report:
Precision: 0.98, Recall: 0.97, F1-score: 0.97

