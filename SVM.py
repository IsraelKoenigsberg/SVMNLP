import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import os

# Load the data from the CSV file
data = pd.read_csv('dataset.csv')

# Define the texts (tweets) and labels (0 = Non-antisemitic, 1 = Antisemitic)
texts = data['Text']
labels = data['Biased']

# Convert the text data into numerical values using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Check if models and vectorizer are already saved
if os.path.exists('svm_model.joblib') and os.path.exists('log_reg_model.joblib') and os.path.exists(
        'vectorizer.joblib') and os.path.exists('random_forest_model.joblib'):
    # Load the models if they are already saved
    svm = load('svm_model.joblib')
    log_reg = load('log_reg_model.joblib')
    rf = load('random_forest_model.joblib')
else:
    # Create and train an SVM classifier
    svm = SVC(kernel='linear', class_weight='balanced')
    svm.fit(X_train, y_train)

    # Create and train a Logistic Regression classifier
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X_train, y_train)

    # Create and train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # Save the models and vectorizer for future use
    dump(svm, 'svm_model.joblib')
    dump(log_reg, 'log_reg_model.joblib')
    dump(vectorizer, 'vectorizer.joblib')
    dump(rf, 'random_forest_model.joblib')

# Create an ensemble classifier using VotingClassifier
ensemble_clf = VotingClassifier(estimators=[('svm', svm), ('log_reg', log_reg), ('rf', rf)], voting='hard')
ensemble_clf.fit(X_train, y_train)

# Make predictions on the test data using all classifiers
y_pred_svm = svm.predict(X_test)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_ensemble = ensemble_clf.predict(X_test)

# Evaluate the performance of the classifiers
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg)}')
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

print(f'Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble)}')
print("Ensemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['(0) Non-antisemitic', '(1) Antisemitic']))

# Function to classify a new text using all classifiers
def classify_text(text):
    # Convert the new text to a TF-IDF vector
    text_vector = vectorizer.transform([text])

    # Get predictions from all classifiers
    pred_svm = svm.predict(text_vector)
    pred_log_reg = log_reg.predict(text_vector)
    pred_rf = rf.predict(text_vector)
    pred_ensemble = ensemble_clf.predict(text_vector)

    # Map the predictions to human-readable labels
    translated_results = {
        'SVM': 'Antisemitic' if pred_svm == 1 else 'Non-antisemitic',
        'Logistic Regression': 'Antisemitic' if pred_log_reg == 1 else 'Non-antisemitic',
        'Random Forest': 'Antisemitic' if pred_rf == 1 else 'Non-antisemitic',
        'Ensemble': 'Antisemitic' if pred_ensemble == 1 else 'Non-antisemitic'
    }

    return translated_results

# Continuously prompt the user to enter text for classification
while True:
    user_input = input("Enter a text to classify (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break

    # Get predictions from all classifiers
    results = classify_text(user_input)

    print("-----------------")
    print(f"Text: {user_input}")
    for clf, result in results.items():
        print(f"{clf} Classification: {result}")
