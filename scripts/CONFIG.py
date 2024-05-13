import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sarcasm_detection.csv')

# SUBMISSION_DATA_TEST_PATH = os.path.join(os.path.dirname(__file__), '../data/test.csv')

OUTPUT_KEY = "is_sarcastic"

TFIDF_OPTIONS = {
    "max_df": 0.8,
    "max_features": 5000
}