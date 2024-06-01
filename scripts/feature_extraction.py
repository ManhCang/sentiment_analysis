# scripts/feature_extraction.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def main():
    # Load the preprocessed dataset
    df = pd.read_csv('../data/IMDB_preprocessed.csv')

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Save the features and labels
    with open('../data/features.pkl', 'wb') as f:
        pickle.dump((X, y), f)
    print("Feature extraction completed and saved to 'features.pkl'")

if __name__ == '__main__':
    main()
