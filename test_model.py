import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataset for testing
    data = pd.read_csv(r'C:\Users\dines\ci-cd-app\data\data.csv')
    data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop('diagnosis', axis=1).reset_index(drop=True)
    y = combined['diagnosis'].reset_index(drop=True)

    # Load the saved model
    model = joblib.load('random_forest_model.joblib')

    # Predict on test data
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy on full dataset: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, predictions))

    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
