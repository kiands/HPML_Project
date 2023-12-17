import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

file_path = 'dataset.csv'  # the file location
data = pd.read_csv(file_path)

# extract text and label
X = data['text']
y = data['output']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use TF-IDF to convert features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# train SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# predict and evaluate
y_pred = svm_model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)

print(report)
