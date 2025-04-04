# app.py

import os
from flask import Flask, request, render_template, redirect, url_for
import joblib
import re
import string # Only needed if using clean_text here

# Initialize Flask app
# Flask automatically looks for templates in a 'templates' folder
# in the same directory as this script.
app = Flask(__name__)

# --- Configuration ---
model_filename = 'spam_model.pkl'
# --- End Configuration ---

# Load the trained pipeline model at startup
pipeline = None
if os.path.exists(model_filename):
    try:
        pipeline = joblib.load(model_filename)
        print(f"Model '{model_filename}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_filename}: {e}")
        # Application might still run but predictions will fail
else:
    print(f"Error: Model file '{model_filename}' not found.")
    print("Please run the train_model.py script first to generate the model.")

# Define the home route - displays the input form
@app.route('/')
def home():
    # Renders the index.html file from the 'templates' folder
    return render_template('index.html')

# Define the prediction route - handles form submission
@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = "Error: Model could not be loaded or used for prediction."
    message_text = ""
    is_spam = None # Use None to indicate prediction hasn't happened or failed

    if pipeline is None:
         # Handle case where model loading failed at startup
         prediction_text = "Model is not available. Please check server logs."
         return render_template('index.html', prediction=prediction_text, message_text=message_text, is_spam=None)

    # Only process if the request method is POST
    if request.method == 'POST':
        message = request.form.get('message', '') # Get message, default to empty string
        message_text = message # Store original message to display back in textarea

        if message: # Check if message is not empty
            try:
                # The pipeline's TF-IDF vectorizer handles necessary text processing
                prediction = pipeline.predict([message]) # Expects a list or iterable
                probability = pipeline.predict_proba([message])

                # Interpret the prediction result (0 = ham, 1 = spam)
                if prediction[0] == 1:
                    spam_prob = probability[0][1] * 100 # Probability of class 1 (spam)
                    prediction_text = f"Prediction: SPAM (Confidence: {spam_prob:.2f}%)"
                    is_spam = True
                    print(f"Prediction: SPAM (Confidence: {spam_prob:.2f}%)")
                else:
                    ham_prob = probability[0][0] * 100 # Probability of class 0 (ham)
                    prediction_text = f"Prediction: HAM / Not Spam (Confidence: {ham_prob:.2f}%)"
                    is_spam = False
                    print(f"Prediction: HAM (Confidence: {ham_prob:.2f}%)")

            except Exception as e:
                print(f"Error during prediction: {e}")
                # Provide a user-friendly error message
                prediction_text = "An error occurred while processing the message."
                is_spam = None # Indicate error state

        else:
            # Handle empty message submission
            prediction_text = "Please enter a message to classify."
            is_spam = None # No prediction made

        # Render the page again, passing the results back to the template
        return render_template('index.html', prediction=prediction_text, message_text=message_text, is_spam=is_spam)

    # Redirect to home page if accessed via GET (shouldn't happen with methods=['POST'])
    return redirect(url_for('home'))

# Run the Flask development server
if __name__ == '__main__':
    # Important for deployment (like Render): listen on 0.0.0.0
    host = '0.0.0.0'
    # Get port from environment variable (used by Render) or default to 5000 for local testing
    port = int(os.environ.get('PORT', 5000))
    # Set debug=False for production/deployment
    # Set debug=True for local development (provides more error details and auto-reloads)
    app.run(host=host, port=port, debug=True)