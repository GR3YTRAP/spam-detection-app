# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import re
import string
import os

print(f"Script starting in directory: {os.getcwd()}")

# --- Configuration ---
data_file = 'email.csv'  # Your data file name
# *** ADJUST HERE: Set these to your actual column names in email.csv ***
text_column = 'text'  # Column containing the email body/message
label_column = 'spam' # Column containing the label (0 for ham, 1 for spam)
# --- End Configuration ---

model_filename = 'spam_model.pkl'

# --- Data Loading ---
print(f"Attempting to load data from: {data_file}")
if not os.path.exists(data_file):
    print(f"Error: File not found at {os.path.abspath(data_file)}")
    print("Please ensure 'email.csv' is in the same directory as this script.")
    exit()

try:
    # Try reading with utf-8 first
    df = pd.read_csv(data_file, encoding='utf-8')
    print("Data loaded successfully (UTF-8 attempt).")
except UnicodeDecodeError:
    print("UTF-8 failed, trying latin-1 encoding...")
    try:
        # Fallback to latin-1 if utf-8 fails
        df = pd.read_csv(data_file, encoding='latin-1')
        print("Data loaded successfully (latin-1 attempt).")
    except Exception as e:
        print(f"Failed to load data with latin-1 encoding as well: {e}")
        exit()
except FileNotFoundError: # Explicitly catch FileNotFoundError
    print(f"Error: The file {data_file} was not found.")
    exit()
except Exception as e: # Catch other potential loading errors
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

print("Columns found:", df.columns.tolist())
# print("First 5 rows:\n", df.head()) # Uncomment to inspect data if needed

# --- Data Validation and Preparation ---
# Check if specified columns exist
if text_column not in df.columns:
    print(f"Error: Text column '{text_column}' not found in the CSV header.")
    print(f"Available columns: {df.columns.tolist()}")
    print("Please update the 'text_column' variable in the script.")
    exit()
if label_column not in df.columns:
    print(f"Error: Label column '{label_column}' not found in the CSV header.")
    print(f"Available columns: {df.columns.tolist()}")
    print("Please update the 'label_column' variable in the script.")
    exit()

print(f"Using column '{text_column}' for messages and '{label_column}' for labels.")
print(f"Label distribution before cleaning:\n{df[label_column].value_counts()}")
print(f"Data shape before cleaning: {df.shape}")

# Drop rows where the essential text or label columns are NaN
initial_rows = df.shape[0]
df.dropna(subset=[text_column, label_column], inplace=True)
rows_after_na_drop = df.shape[0]
if initial_rows > rows_after_na_drop:
    print(f"Dropped {initial_rows - rows_after_na_drop} rows due to NaNs in essential columns.")
print(f"Data shape after dropping NaNs: {df.shape}")


# --- Text Cleaning Function ---
def clean_text(text):
    """Cleans text data: lowercase, remove brackets, links, HTML, punctuation, newlines, words with numbers."""
    if not isinstance(text, str):
        text = str(text) # Convert potential non-strings just in case
    text = text.lower()
    # Use raw strings (r'...') for regex patterns to avoid SyntaxWarnings
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove links
    text = re.sub(r'<.*?>+', '', text) # Remove html tags
    # Use re.escape on punctuation for safety within character sets
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text) # Replace newline with space instead of removing entirely
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    # Optional: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Applying text cleaning...")
# Apply cleaning to the specified text column
df['message_clean'] = df[text_column].apply(clean_text)
print("Text cleaning complete.")


# --- Label Handling ---
print(f"Data type of label column '{label_column}': {df[label_column].dtype}")
unique_labels = df[label_column].unique()
print(f"Unique values in label column '{label_column}' (showing up to 20): {unique_labels[:20]}")

# Prepare X (features) and y (target)
X = df['message_clean'] # Use the newly cleaned text column
y = df[label_column]    # Use the original label column

# Basic check if labels look like standard 0/1 format
if not pd.api.types.is_numeric_dtype(y) or not all(y.isin([0, 1])):
     print(f"Warning: Label column '{label_column}' may contain non-numeric or values other than 0 and 1.")
     # *** ADJUST HERE (Optional but Recommended): If labels are 'spam'/'ham', explicitly map them ***
     if y.dtype == 'object': # Check if labels are strings
         print("Labels appear to be strings. Attempting common mapping {'ham': 0, 'spam': 1}...")
         # Make sure the keys match your actual labels exactly (case-sensitive)
         label_map = {'ham': 0, 'spam': 1}
         y_mapped = y.map(label_map)

         # Check if any labels were NOT mapped (resulted in NaN)
         unmapped_count = y_mapped.isnull().sum()
         if unmapped_count > 0:
             print(f"Warning: Found {unmapped_count} labels that did not match 'ham' or 'spam'. These rows will be dropped.")
             # Keep only rows where mapping was successful
             X = X[y_mapped.notnull()]
             y = y_mapped.dropna()
         else:
             y = y_mapped
             print("String label mapping successful.")
     else:
         print("Labels are not 0/1 and not standard strings. Manual review and mapping might be required.")
         # Consider adding more specific mapping logic if needed, or exiting if labels are unexpected.
         # exit() # Uncomment to stop if labels are unexpected
else:
    print("Labels appear to be numeric and contain only 0s and 1s.")

# Ensure y is integer type after potential mapping
try:
    y = y.astype(int)
    print("Target variable 'y' successfully converted to integer type.")
except ValueError as e:
    print(f"Error converting labels to integers after processing: {e}.")
    print("Please check the content of the label column and the mapping logic.")
    exit()
except Exception as e: # Catch any other conversion error
    print(f"Unexpected error converting labels to integers: {e}")
    exit()


# Check if data remains after processing
if X.empty or y.empty:
    print("Error: No data left after processing/cleaning/mapping. Cannot proceed.")
    print("Please check your input data, column names, and cleaning/mapping logic.")
    exit()


# --- Data Splitting ---
print(f"Final data shape for splitting: X={X.shape}, y={y.shape}")
# Check if there are enough samples in each class for stratification
min_class_count = y.value_counts().min()
if min_class_count < 2:
    print(f"Warning: The smallest class ('{y.value_counts().idxmin()}') has only {min_class_count} sample(s). Stratification may fail or be unreliable.")
    print("Consider disabling stratification if errors occur, or getting more data for the minority class.")
    # Use stratify=None if min_class_count < 2 and test_size > 0
    stratify_option = None if min_class_count < 2 else y
else:
    stratify_option = y

try:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 20% for testing
        random_state=42,      # For reproducible results
        stratify=stratify_option # Try to maintain class balance
    )
    print(f"Data split into Train ({X_train.shape[0]} samples) and Test ({X_test.shape[0]} samples).")
    print(f"Train label distribution (Normalized):\n{y_train.value_counts(normalize=True)}")
    print(f"Test label distribution (Normalized):\n{y_test.value_counts(normalize=True)}")
except ValueError as e:
     print(f"Error during train_test_split: {e}")
     print("This often happens if a class has too few samples for the requested test_size and stratification.")
     print("Try adjusting test_size or check data processing steps.")
     exit()
except Exception as e:
    print(f"Unexpected error during train_test_split: {e}")
    exit()


# --- Model Building (Pipeline) ---
# Create a pipeline: transforms data using TF-IDF, then classifies using Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), # Convert text to TF-IDF features, remove common English words
    ('clf', MultinomialNB())                       # Classifier suitable for text data
])
print("Pipeline created (TF-IDF Vectorizer -> MultinomialNB Classifier).")

# --- Model Training ---
print("Training the model...")
try:
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
except Exception as e:
    print(f"An error occurred during model training: {e}")
    exit()

# --- Model Evaluation ---
try:
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
except Exception as e:
    print(f"An error occurred during model evaluation: {e}")
    # Continue to save the model even if evaluation fails, but log the error.

# --- Save the Pipeline ---
try:
    joblib.dump(pipeline, model_filename)
    print(f"Model pipeline saved successfully as {model_filename}")
except Exception as e:
    print(f"Error saving the model pipeline to {model_filename}: {e}")
    exit() # Exit if model saving fails

print("Training script finished.")