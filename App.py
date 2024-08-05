import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def analyze_sentiment_vader(text):
    """Analyze the sentiment of a given text using VADER."""
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment = sia.polarity_scores(text)

    # Display results
    print(f"Positive: {sentiment['pos']:.2f}")
    print(f"Neutral: {sentiment['neu']:.2f}")
    print(f"Negative: {sentiment['neg']:.2f}")
    print(f"Compound: {sentiment['compound']:.2f}")


def main():
    nltk.download('vader_lexicon')  # Download VADER lexicon
    text = input("Enter a sentence to analyze: ")
    analyze_sentiment_vader(text)


if __name__ == "__main__":
    main()
