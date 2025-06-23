import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("app_feedback_sentiment_dataset.csv")  # or use pd.read_csv("yourfile.csv")

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize - BoW
bow_vectorizer = CountVectorizer(stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Vectorize - TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train BoW model
bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)
y_pred_bow = bow_model.predict(X_test_bow)

# Train TF-IDF model
tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

# Evaluation
bow_acc = accuracy_score(y_test, y_pred_bow)
tfidf_acc = accuracy_score(y_test, y_pred_tfidf)

bow_report = classification_report(y_test, y_pred_bow, output_dict=True)
tfidf_report = classification_report(y_test, y_pred_tfidf, output_dict=True)

print("📦 Bag-of-Words Accuracy:", bow_acc)
print(classification_report(y_test, y_pred_bow))

print("\n📘 TF-IDF Accuracy:", tfidf_acc)
print(classification_report(y_test, y_pred_tfidf))

# ================================
# 📊 Comparison Graph (Accuracy & F1)
# ================================
methods = ['Bag-of-Words', 'TF-IDF']
accuracies = [bow_acc, tfidf_acc]
f1_scores = [bow_report['weighted avg']['f1-score'], tfidf_report['weighted avg']['f1-score']]

x = range(len(methods))
plt.figure(figsize=(8, 5))
plt.bar(x, accuracies, width=0.4, label='Accuracy', align='center', color='skyblue')
plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1 Score', align='center', color='orange')
plt.xticks([i + 0.2 for i in x], methods)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Comparison of BoW vs TF-IDF (Accuracy & F1 Score)")
plt.legend()
plt.tight_layout()
plt.show()
