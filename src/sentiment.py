import csv
import asyncio
import json
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec,  KeyedVectors
import gensim.downloader as api
import numpy as np

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add checkpoint file path
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, 'data', 'sentiment_checkpoint.json')

input_csv_path = os.path.join(PROJECT_ROOT, 'data', 'reviews.csv')
output_csv_path = os.path.join(PROJECT_ROOT, 'data', 'review_sentiments.csv')

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class Review:
    def __init__(self, review_data):
        """
        Initialize a Review object from an array with the structure:
        [rating, reviewerName, reviewText, categories, reviewTime, unixReviewTime, 
         formattedDate, gPlusPlaceId, gPlusUserId]
        """
        self.rating = review_data[0]
        self.reviewer_name = review_data[1]
        self.review_text = review_data[2]
        self.categories = review_data[3]
        self.review_time = review_data[4]
        self.unix_review_time = review_data[5]
        self.formatted_date = review_data[6]
        self.gplus_place_id = review_data[7]
        self.gplus_user_id = review_data[8]

class ReviewSentiment:
    def __init__(self, review_id, overall_score, overall_magnitude, food_score, 
                 service_score, value_score, ambiance_score, emotions):
        self.review_id = review_id
        self.overall_score = overall_score
        self.overall_magnitude = overall_magnitude
        self.food_score = food_score
        self.service_score = service_score
        self.value_score = value_score
        self.ambiance_score = ambiance_score
        self.emotions = emotions  # JSON string of emotions

def get_processed_reviews():
    """
    Read the checkpoint file to get already processed review IDs
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_ids, sentiment_results):
    """
    Save both processed review IDs and their sentiment results to checkpoint files
    """
    # Save processed IDs
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(processed_ids), f)
    
    # Save sentiment results
    results_checkpoint = os.path.join(PROJECT_ROOT, 'data', 'sentiment_results_checkpoint.json')
    results_to_save = []
    for result in sentiment_results:
        results_to_save.append({
            'review_id': result.review_id,
            'user_id': result.user_id,
            'restaurant_id': result.restaurant_id,
            'overall_score': result.overall_score,
            'overall_magnitude': result.overall_magnitude,
            'food_score': result.food_score,
            'service_score': result.service_score,
            'value_score': result.value_score,
            'ambiance_score': result.ambiance_score,
            'emotions': result.emotions
        })
    
    with open(results_checkpoint, 'w') as f:
        json.dump(results_to_save, f)

def load_checkpoint():
    """
    Load both processed IDs and sentiment results from checkpoint files
    """
    processed_ids = set()
    sentiment_results = []
    
    # Load processed IDs
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed_ids = set(json.load(f))
    
    # Load sentiment results
    results_checkpoint = os.path.join(PROJECT_ROOT, 'data', 'sentiment_results_checkpoint.json')
    if os.path.exists(results_checkpoint):
        with open(results_checkpoint, 'r') as f:
            saved_results = json.load(f)
            for result_dict in saved_results:
                sentiment = ReviewSentiment(
                    review_id=result_dict['review_id'],
                    overall_score=result_dict['overall_score'],
                    overall_magnitude=result_dict['overall_magnitude'],
                    food_score=result_dict['food_score'],
                    service_score=result_dict['service_score'],
                    value_score=result_dict['value_score'],
                    ambiance_score=result_dict['ambiance_score'],
                    emotions=result_dict['emotions']
                )
                sentiment.user_id = result_dict['user_id']
                sentiment.restaurant_id = result_dict['restaurant_id']
                sentiment_results.append(sentiment)
    
    return processed_ids, sentiment_results

def write_sentiment_results_to_csv(sentiments, output_file_path, append=False):
    """
    Write sentiment analysis results to a CSV file
    """
    mode = 'a' if append else 'w'
    write_header = not (append and os.path.exists(output_file_path))
    
    with open(output_file_path, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header only if new file or not appending
        if write_header:
            writer.writerow([
                'review_id', 'user_id', 'restaurant_id', 'overall_score', 
                'overall_magnitude', 'food_score', 'service_score', 
                'value_score', 'ambiance_score', 'emotions'
            ])
        
        # Write data
        for sentiment in sentiments:
            writer.writerow([
                sentiment.review_id,
                sentiment.user_id,
                sentiment.restaurant_id,
                sentiment.overall_score,
                sentiment.overall_magnitude,
                sentiment.food_score,
                sentiment.service_score,
                sentiment.value_score,
                sentiment.ambiance_score,
                sentiment.emotions
            ])

def load_word2vec_model():
    """
    Load pre-trained Word2Vec model
    You can use different pre-trained models like:
    - Google News vectors (300d)
    - GloVe vectors
    - FastText vectors
    """
    # Example using Google's pre-trained model
    model_path = 'path/to/GoogleNews-vectors-negative300.bin'
    return api.load('word2vec-google-news-300')

def preprocess_text(text):
    """
    Preprocess text by:
    1. Tokenizing
    2. Converting to lowercase
    3. Removing stopwords
    4. Lemmatizing
    """
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Get POS tags for better lemmatization
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatize with POS tags
    lemmatized = []
    for token, pos in pos_tags:
        pos_char = pos[0].lower()
        if pos_char in ['a', 'r', 'n', 'v']:  # adj, adv, noun, verb
            pos_tag = {'a': 'a', 'r': 'r', 'n': 'n', 'v': 'v'}[pos_char]
            lemmatized.append(lemmatizer.lemmatize(token, pos=pos_tag))
        else:
            lemmatized.append(lemmatizer.lemmatize(token))
    
    return lemmatized

# Define aspect seed words (lemmatized)
ASPECT_SEEDS = {
    'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'menu', 'ingredient', 'cook'],
    'service': ['service', 'staff', 'waiter', 'server', 'hospitality', 'attend'],
    'value': ['price', 'value', 'cost', 'worth', 'expensive', 'cheap', 'afford'],
    'ambiance': ['ambiance', 'atmosphere', 'decor', 'environment', 'setting', 'mood']
}

def get_aspect_vectors(word2vec_model, aspect_seeds):
    """
    Calculate average vector for each aspect using seed words
    """
    aspect_vectors = {}
    
    for aspect, seeds in aspect_seeds.items():
        vectors = []
        for seed in seeds:
            if seed in word2vec_model:
                vectors.append(word2vec_model[seed])
        
        if vectors:
            aspect_vectors[aspect] = np.mean(vectors, axis=0)
    
    return aspect_vectors

def analyze_aspect_with_word2vec(tokens, word2vec_model, aspect_vectors, similarity_threshold=0.4):
    """
    Analyze aspects using Word2Vec similarity
    Returns dictionary of aspects with their confidence scores
    """
    aspects_detected = {aspect: [] for aspect in aspect_vectors.keys()}
    
    for token in tokens:
        if token in word2vec_model:
            token_vector = word2vec_model[token]
            
            # Calculate similarity with each aspect
            for aspect, aspect_vector in aspect_vectors.items():
                similarity = np.dot(token_vector, aspect_vector) / (
                    np.linalg.norm(token_vector) * np.linalg.norm(aspect_vector)
                )
                
                if similarity > similarity_threshold:
                    aspects_detected[aspect].append(similarity)
    
    # Calculate average confidence for each detected aspect
    return {
        aspect: np.mean(scores) if scores else 0.0 
        for aspect, scores in aspects_detected.items()
    }

def analyze_review_sentiment(review_arrays):
    """
    Analyze sentiment in restaurant reviews using NLTK/VADER with Word2Vec-enhanced aspect detection
    """
    processed_reviews, previous_results = load_checkpoint()
    results = previous_results.copy()  # Start with previously processed results
    newly_processed = set()
    
    # Initialize models and tools
    sia = SentimentIntensityAnalyzer()
    word2vec_model = load_word2vec_model()
    aspect_vectors = get_aspect_vectors(word2vec_model, ASPECT_SEEDS)
    
    # Convert review arrays to Review objects and filter out already processed ones
    reviews = []
    for review_array in review_arrays:
        review = Review(review_array)
        review_id = f"{review.gplus_user_id}_{review.gplus_place_id}"
        if review_id not in processed_reviews:
            reviews.append(review)
    
    print(f"Analyzing sentiment for {len(reviews)} remaining reviews...")
    
    try:
        for i, review in enumerate(reviews):
            print(f"{i}/{len(reviews)} reviews completed")
            try:
                if not review.review_text or review.review_text.strip() == "":
                    continue
                
                # Analyze overall sentiment using VADER
                scores = sia.polarity_scores(review.review_text)
                overall_score = scores['compound']
                overall_magnitude = abs(scores['pos'] - scores['neg'])
                
                # Split review into sentences
                sentences = nltk.sent_tokenize(review.review_text)
                
                aspect_scores = {
                    'food': [],
                    'service': [],
                    'value': [],
                    'ambiance': []
                }
                
                for sentence in sentences:
                    # Preprocess the sentence
                    tokens = preprocess_text(sentence)
                    
                    # Get sentiment scores
                    sent_scores = sia.polarity_scores(sentence)
                    sentiment_score = sent_scores['compound']
                    
                    # Detect aspects using Word2Vec
                    aspect_confidences = analyze_aspect_with_word2vec(
                        tokens, 
                        word2vec_model, 
                        aspect_vectors
                    )
                    
                    # Add weighted sentiment scores for each aspect
                    for aspect, confidence in aspect_confidences.items():
                        if confidence > 0:
                            aspect_scores[aspect].append(sentiment_score * confidence)
                
                # Map emotions based on compound score and magnitude
                emotions = []
                if overall_score >= 0.5:
                    emotions.append("joy" if overall_magnitude > 0.5 else "satisfaction")
                elif overall_score >= 0.1:
                    emotions.append("contentment")
                elif overall_score <= -0.5:
                    emotions.append("anger" if overall_magnitude > 0.5 else "frustration")
                elif overall_score <= -0.1:
                    emotions.append("disappointment")
                
                # Calculate final aspect scores
                final_scores = {}
                for aspect, scores in aspect_scores.items():
                    final_scores[aspect] = sum(scores) / len(scores) if scores else 0.0
                
                result = ReviewSentiment(
                    review_id=f"{review.gplus_user_id}_{review.gplus_place_id}",
                    overall_score=overall_score,
                    overall_magnitude=overall_magnitude,
                    food_score=final_scores['food'],
                    service_score=final_scores['service'],
                    value_score=final_scores['value'],
                    ambiance_score=final_scores['ambiance'],
                    emotions=emotions
                )
                
                result.user_id = review.gplus_user_id
                result.restaurant_id = review.gplus_place_id
                
                results.append(result)
                newly_processed.add(result.review_id)
                
                # Save checkpoint every 100 reviews
                if i > 0 and i % 100 == 0:
                    all_processed = processed_reviews.union(newly_processed)
                    save_checkpoint(all_processed, results)
                    print(f"Checkpoint saved at review {i}")
                
            except Exception as error:
                print(f"Error analyzing review: {error}")
                continue
        
        # Save final checkpoint
        all_processed = processed_reviews.union(newly_processed)
        save_checkpoint(all_processed, results)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving checkpoint before exit...")
        all_processed = processed_reviews.union(newly_processed)
        save_checkpoint(all_processed, results)
        print("Checkpoint saved. You can resume later from where you left off.")
        return results
    
    return results

def read_reviews_from_csv(csv_file_path):
    """
    Read reviews from a CSV file
    
    The CSV file should have columns in this order:
    rating, reviewerName, reviewText, categories, reviewTime, unixReviewTime, 
    formattedDate, gPlusPlaceId, gPlusUserId
    """
    reviews = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Skip header row if it exists
        header = next(csv_reader, None)
        
        for row in csv_reader:
            if len(row) >= 9:  # Ensure we have all required fields
                reviews.append(row)
    
    return reviews

def main():
    # Read reviews from CSV
    print(f"Reading reviews from {input_csv_path}...")
    reviews = read_reviews_from_csv(input_csv_path)
    print(f"Read {len(reviews)} reviews.")
    
    try:
        # Analyze sentiments
        sentiments = analyze_review_sentiment(reviews)
        print(f"Analyzed sentiment for {len(sentiments)} reviews.")
        
        # Write results to CSV
        write_sentiment_results_to_csv(sentiments, output_csv_path)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return
    
    print(f"Sentiment analysis completed. Results written to {output_csv_path}")

# Run the main function
if __name__ == "__main__":
    main()