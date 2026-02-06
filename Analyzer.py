import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Load and analyze data
data = pd.read_csv("customer_feedback.csv")
print(f"Loaded {len(data)} reviews")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zs]', '', text)
    stop_words = ['the', 'is', 'are', 'was', 'and', 'or', 'but', 'in', 'on']
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

data['cleaned_review'] = data['review'].apply(clean_text)

# Sentiment analysis
positive_words = ['good', 'excellent', 'great', 'amazing', 'love', 'happy', 'satisfied']
negative_words = ['bad', 'poor', 'terrible', 'awful', 'late', 'slow', 'rude', 'broken']

def analyze_sentiment(text):
    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)
    return "Positive" if pos_score > neg_score else "Negative" if neg_score > pos_score else "Neutral"

data['sentiment'] = data['cleaned_review'].apply(analyze_sentiment)

# Save results
data.to_csv('results.csv', index=False)

# Create charts
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Distribution')

# Top issues
plt.subplot(1,2,2)
neg_reviews = ' '.join(data[data['sentiment']=='Negative']['cleaned_review'])
from wordcloud import WordCloud
wc = WordCloud().generate(neg_reviews)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Common Issues')
plt.savefig('analysis_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Check results.csv and analysis_charts.png")
