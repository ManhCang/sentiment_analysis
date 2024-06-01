# scripts/data_preprocessing.py
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data files (if not already installed)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def main():
    # Load the dataset
    df = pd.read_csv('data/IMDB Dataset.csv')

    # Apply preprocessing to the reviews
    df['review'] = df['review'].apply(preprocess_text)

    # Save the preprocessed data
    df.to_csv('data/IMDB_preprocessed.csv', index=False)
    print("Data preprocessing completed and saved to 'IMDB_preprocessed.csv'")

if __name__ == '__main__':
    main()
