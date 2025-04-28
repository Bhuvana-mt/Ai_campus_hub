import csv
import random
import nltk
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
spell = SpellChecker()

app = Flask(__name__)

intent_classifier = None
vectorizer = None
dataset_path = "dataset/college_chatbot_dataset.csv"


def preprocess_text(text):
    # Normalize text to lowercase, remove punctuation, and handle common abbreviations
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(word) if word not in stop_words else word for word in tokens if word.isalpha()]
    return " ".join(tokens)


def load_dataset(file_path):
    data = []
    labels = []
    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(preprocess_text(row["Question"].strip()))
                labels.append(row["Response"].strip())
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return data, labels


def train_intent_classifier():
    global intent_classifier, vectorizer

    data, labels = load_dataset(dataset_path)
    if not data or not labels:
        print("Dataset is empty or missing!")
        return

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    intent_classifier = LogisticRegression()
    intent_classifier.fit(X_train_vectors, y_train)

    y_pred = intent_classifier.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    joblib.dump(intent_classifier, "intent_classifier.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


def load_ml_model():
    global intent_classifier, vectorizer
    try:
        intent_classifier = joblib.load("intent_classifier.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("Model files not found! Training a new model...")
        train_intent_classifier()


def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return " ".join(corrected_words)


def load_pairs():
    pairs = []
    data, labels = load_dataset(dataset_path)
    print(f"Loaded data: {data[:5]}")  # Debug: print first 5 data points
    print(f"Loaded labels: {labels[:5]}")  # Debug: print first 5 labels
    for question, response in zip(data, labels):
        pairs.append((question, [response]))
    return pairs


pairs = load_pairs()


def get_response_from_dataset(user_input, pairs, threshold=60):
    best_match_score = 0
    best_match_response = "I'm sorry, I didn't understand that. Can you please rephrase?"

    # First, check for an exact match for names with additional normalization
    for pattern, responses in pairs:
        normalized_input = normalize_name(user_input)
        normalized_pattern = normalize_name(pattern)

        if normalized_input == normalized_pattern:
            best_match_response = random.choice(responses)
            return best_match_response  # Return immediately if there's an exact match

    # If no exact match, perform fuzzy matching with a threshold
    for pattern, responses in pairs:
        match_score = fuzz.partial_ratio(user_input.lower(), pattern.lower())  # Using partial_ratio for better results
        print(f"Matching '{user_input}' with '{pattern}', score: {match_score}")  # Debug the match scores
        if match_score > best_match_score and match_score >= threshold:
            best_match_score = match_score
            best_match_response = random.choice(responses)

    # If no good match was found, predict using the model
    if best_match_score < threshold:
        print("No match found in dataset, falling back to model...")
        best_match_response = intent_classifier.predict(vectorizer.transform([user_input]))[0]
        print(f"Model Prediction: {best_match_response}")  # Debug: Log model prediction

    return best_match_response


def normalize_name(name):
    """
    Normalize the name by removing unwanted characters, standardizing case, and trimming spaces.
    """
    name = name.lower().strip()
    name = ''.join(e for e in name if e.isalnum() or e.isspace())  # Remove punctuation and special characters
    return name


def get_closest_match(user_input, pairs):
    # Extract the list of questions from the pairs dataset
    questions = [pair[0] for pair in pairs]

    # Create a TfidfVectorizer to convert the questions into TF-IDF vectors
    vectorizer = TfidfVectorizer()

    # Combine user input with the dataset's questions
    all_questions = questions + [user_input]
    
    # Transform the questions into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    # Compute the cosine similarity between the user input (last element) and the dataset's questions
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the index of the most similar question
    closest_match_index = similarity_scores.argmax()

    # Return the closest match question and its corresponding response
    closest_match = questions[closest_match_index]
    response = pairs[closest_match_index][1][0]  # Accessing response from the tuple

    return closest_match, response


def get_related_questions_with_responses(user_input, pairs, top_n=3):
    """
    Find related questions and their responses based on similarity to user input.
    """
    related_questions = []
    for pattern, responses in pairs:
        similarity = fuzz.partial_ratio(user_input.lower(), pattern.lower())
        related_questions.append((similarity, pattern, responses[0]))
    
    # Sort by similarity and pick top N
    related_questions = sorted(related_questions, key=lambda x: x[0], reverse=True)[:top_n]
    return [{"question": q[1], "response": q[2]} for q in related_questions]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    print(f"User Input: {user_input}")  # Debugging: Log the user input

    # Correct spelling and preprocess
    corrected_input = correct_spelling(user_input)
    print(f"Corrected Input: {corrected_input}")  # Debugging: Log the corrected input

    processed_input = preprocess_text(corrected_input)
    print(f"Processed Input: {processed_input}")  # Debugging: Log the processed input

    # Fetch response from dataset based on processed input
    response = get_response_from_dataset(processed_input, pairs)
    print(f"Response from Dataset: {response}")  # Debugging: Log the response from dataset

    # If no response was found, fallback to model prediction
    if response == "I'm sorry, I didn't understand that. Can you please rephrase?":
        print("Trying model prediction...")  # Debugging: Log when fallback is used
        response = intent_classifier.predict(vectorizer.transform([processed_input]))[0]
        print(f"Model Prediction Response: {response}")  # Debugging: Log the model's response

    # Fetch related questions and their responses from the dataset
    related = get_related_questions_with_responses(processed_input, pairs)
    print(f"Related Questions and Responses: {related}")  # Debugging: Log related questions

    # Log the closest match (optional, for debugging purposes)
    closest_match, matched_response = get_closest_match(processed_input, pairs)
    print(f"Matched Question: {closest_match}")  # Debugging: Log the closest match found

    # Return the response and related questions to the frontend
    return jsonify({"response": response, "related": related, "closest_match": matched_response})


if __name__ == "__main__":
    load_ml_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
