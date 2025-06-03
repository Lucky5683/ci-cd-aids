import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # Load your dataset
    data = pd.read_csv(r'C:\Users\dines\ci-cd-app\data\data.csv')

    # Drop unnecessary columns
    data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

    # Separate features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Convert target labels to numeric (M=1, B=0)
    y = y.map({'M': 1, 'B': 0})

    # Combine X and y to drop any rows with NaNs
    combined = pd.concat([X, y], axis=1)
    initial_len = len(combined)
    combined = combined.dropna()
    after_drop_len = len(combined)
    print(f"Dropped {initial_len - after_drop_len} rows with NaN values in features or target.")

    # Split back to X and y after dropping NaNs
    X = combined.drop('diagnosis', axis=1).reset_index(drop=True)
    y = combined['diagnosis'].reset_index(drop=True)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for NaNs after split
    print("NaNs in y_train:", y_train.isna().sum())
    print("NaNs in X_train:", X_train.isna().sum().sum())

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Predict on test data
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on test data: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Feature importance
    importances = model.feature_importances_
    features = X.columns
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    feat_imp.head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importances')
    plt.show()

    # Save the model
    
    print("Model saved as random_forest_model.joblib")
    joblib.dump(model, 'random_forest_model.joblib')

if __name__ == "__main__":
    main()

