# scripts/model_training.py
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load the features and labels
    with open('../data/features.pkl', 'rb') as f:
        X, y = pickle.load(f)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save the trained model
    with open('../data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model training completed and saved to 'model.pkl'")

if __name__ == '__main__':
    main()
