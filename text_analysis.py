import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import numpy as np

# Input sentences
sentences = [
    "Sam eats pizza after football.",
    "Pizza and burgers are delicious.",
    "Devi plays football on Sunday.",
    "Burgers and pizza after game.",
    "She loves pizza and tennis."
]

# Vectorizers
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

# Transform data
X_bow = bow_vectorizer.fit_transform(sentences)
X_tfidf = tfidf_vectorizer.fit_transform(sentences)

bow_features = bow_vectorizer.get_feature_names_out()
tfidf_features = tfidf_vectorizer.get_feature_names_out()

# -------------------------------
# 🔵 1. BoW – Individual Word Clouds
# -------------------------------
print("Showing BoW Word Clouds (Individual Sentences)...")

for i, row in enumerate(X_bow.toarray()):
    word_freq = {word: freq for word, freq in zip(bow_features, row) if freq > 0}
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(6,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"BoW - Sentence {i+1}")
    plt.show(block=True)

# -------------------------------
# 🔵 2. BoW – Combined Word Cloud
# -------------------------------
print("Showing BoW Word Cloud (All Sentences Combined)...")

combined_bow = np.sum(X_bow.toarray(), axis=0)
combined_word_freq = dict(zip(bow_features, combined_bow))
wordcloud_combined_bow = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(combined_word_freq)

plt.figure(figsize=(6,4))
plt.imshow(wordcloud_combined_bow, interpolation='bilinear')
plt.axis('off')
plt.title("BoW - All Sentences Combined")
plt.show(block=True)

# -------------------------------
# 🔵 3. BoW – Cosine Similarity Heatmap
# -------------------------------
sim_bow = cosine_similarity(X_bow)

plt.figure(figsize=(7,5))
sns.heatmap(sim_bow, annot=True, cmap='YlGnBu',
            xticklabels=[f"S{i+1}" for i in range(5)],
            yticklabels=[f"S{i+1}" for i in range(5)])
plt.title("BoW - Document Similarity Matrix")
plt.show(block=True)

# -------------------------------
# 🟡 4. TF-IDF – Individual Word Clouds
# -------------------------------
print("Showing TF-IDF Word Clouds (Individual Sentences)...")

for i, row in enumerate(X_tfidf.toarray()):
    word_score = {word: score for word, score in zip(tfidf_features, row) if score > 0}
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_score)
    
    plt.figure(figsize=(6,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"TF-IDF - Sentence {i+1}")
    plt.show(block=True)

# -------------------------------
# 🟡 5. TF-IDF – Combined Word Cloud
# -------------------------------
print("Showing TF-IDF Word Cloud (All Sentences Combined)...")

combined_tfidf = np.sum(X_tfidf.toarray(), axis=0)
combined_word_score = dict(zip(tfidf_features, combined_tfidf))
wordcloud_combined_tfidf = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(combined_word_score)

plt.figure(figsize=(6,4))
plt.imshow(wordcloud_combined_tfidf, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF - All Sentences Combined")
plt.show(block=True)

# -------------------------------
# 🟡 6. TF-IDF – Cosine Similarity Heatmap
# -------------------------------
sim_tfidf = cosine_similarity(X_tfidf)

plt.figure(figsize=(7,5))
sns.heatmap(sim_tfidf, annot=True, cmap='YlOrRd',
            xticklabels=[f"S{i+1}" for i in range(5)],
            yticklabels=[f"S{i+1}" for i in range(5)])
plt.title("TF-IDF - Document Similarity Matrix")
plt.show(block=True)

# Done
input("Press Enter to exit...")
