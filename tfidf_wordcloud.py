from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample documents
docs = [
    "Machine learning is fun",
    "Learning machines can be interesting",
    "Fun with machine learning and AI"
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
features = vectorizer.get_feature_names_out()

# Create Word Cloud based on TF-IDF weights
word_freq = dict(zip(features, X.toarray().sum(axis=0)))
wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_freq)

# Show Word Cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("TF-IDF Word Cloud")
plt.show()

# Cosine Similarity Matrix
sim_matrix = cosine_similarity(X)
print("TF-IDF Cosine Similarity Matrix:\n", sim_matrix)
