import pandas as pd  # Importing pandas for data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical representation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression
from sklearn.ensemble import VotingClassifier  # Importing Voting Classifier for ensemble method
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation metrics
from joblib import dump, load  # For saving and loading models
import os  # For checking if files exist

# Load the data from the CSV file
data = pd.read_csv('dataset.csv')

# Define the texts (tweets) and labels (0 = Non-antisemitic, 1 = Antisemitic)
texts = data['Text']
labels = data['Biased']

# Check if models and vectorizer are already saved
if os.path.exists('svm_model.joblib') and os.path.exists('log_reg_model.joblib') and os.path.exists(
        'vectorizer.joblib'):
    # Load the models and vectorizer if they are already saved

    svm = load('svm_model.joblib')
    log_reg = load('log_reg_model.joblib')
    vectorizer = load('vectorizer.joblib')
else:
    # Convert the text data into numerical values using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(texts)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Create and train an SVM classifier
    svm = SVC(kernel='linear', class_weight='balanced')  # Uses balanced class weight since antisemitic text is the
    # great minority
    svm.fit(X_train, y_train)

    # Create and train a Logistic Regression classifier
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X_train, y_train)

    # Save the models and vectorizer for future use
    dump(svm, 'svm_model.joblib')
    dump(log_reg, 'log_reg_model.joblib')
    dump(vectorizer, 'vectorizer.joblib')

# Convert the text data into numerical values using the loaded vectorizer
X = vectorizer.transform(texts)

# Create an ensemble classifier using VotingClassifier
ensemble_clf = VotingClassifier(estimators=[('svm', svm), ('log_reg', log_reg)], voting='hard')
ensemble_clf.fit(X, labels)

# Split the data into training and testing sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Make predictions on the test data using SVM
y_pred_svm = svm.predict(X_test)

# Make predictions on the test data using Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# Make predictions on the test data using Ensemble Classifier
y_pred_ensemble = ensemble_clf.predict(X_test)

# Evaluate the performance of the SVM model
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')

# Evaluate the performance of the Logistic Regression model
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg)}')

# Evaluate the performance of the Ensemble model
print(f'Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble)}')

# Classification report for SVM
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

# Classification report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

# Classification report for Ensemble model
print("Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))


# Function to classify a new text using the ensemble classifier
def classify_text_ensemble(text):
    # Convert the new text to a TF-IDF vector
    text_vector = vectorizer.transform([text])
    # Predict the label using the ensemble classifier
    prediction = ensemble_clf.predict(text_vector)
    # Map the prediction to a human-readable label
    label = 'Antisemitic' if prediction == 1 else 'Non-antisemitic'
    return label


# Continuously prompt the user to enter text for classification
while True:
    user_input = input("Enter a text to classify (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    # Get prediction from the ensemble classifier
    result_ensemble = classify_text_ensemble(user_input)

    print("-----------------")
    print(f"Text: {user_input} | Ensemble Classification: {result_ensemble}")
