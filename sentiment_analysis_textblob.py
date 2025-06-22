from textblob import TextBlob

# Sample text
text = "I love the new AI features. They are really helpful and make my work easier!"

# Create a TextBlob object
blob = TextBlob(text)

# Perform sentiment analysis
sentiment = blob.sentiment

print("Polarity:", sentiment.polarity)
print("Subjectivity:", sentiment.subjectivity)

texts = [
    "I absolutely love this!",
    "It's okay, not great.",
    "I hate this experience."
]

for sentence in texts:
    blob = TextBlob(sentence)
    print(f"\nText: {sentence}")
    print(f"Polarity: {blob.sentiment.polarity}")
    print(f"Subjectivity: {blob.sentiment.subjectivity}")