# NLP & Sequence Modelling Mini Project

## Project Overview
This project builds a sentiment classification pipeline for customer support messages. It uses both traditional NLP models and a sequence model to classify text into three sentiment classes: `positive`, `neutral`, and `negative`.

## Dataset
- Source file: `customer_support_text_classification.csv`
- Data source: [Google Drive folder](https://drive.google.com/drive/folders/16lHceA9Y0e3BMD6Ru3sOl7PP7yO2_s50?usp=drive_link)
- Main columns:
  - `customer_message` — the raw text of a customer support message
  - `sentiment_label` — the target sentiment label
- The notebook performs exploratory analysis on dataset size, label balance, sample text, and text length.

## Approach
1. Load the dataset and inspect data quality, missing values, and class distribution.
2. Clean and preprocess the text data before feature extraction.
3. Create text representations using Bag-of-Words and TF-IDF.
4. Train baseline classifiers:
   - Logistic Regression
   - Multinomial Naive Bayes
5. Build a sequence model using an LSTM-based architecture for text classification.

## Steps
1. Setup and imports
   - Load required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and optional `tensorflow` for the LSTM model.
2. Dataset understanding
   - Read the CSV file
   - Display dataset shape, data types, missing values, label frequencies, and sample messages
3. Text preprocessing
   - Normalize and clean text using regex and text-cleaning utilities
   - Create a cleaned text column for vectorization and sequence encoding
4. Feature engineering
   - Build `CountVectorizer` and `TfidfVectorizer` representations
   - Train/test split the data for model evaluation
5. Classification models
   - Train Logistic Regression and Multinomial Naive Bayes on text features
   - Evaluate using accuracy, classification reports, and confusion matrices
6. Sequence model
   - Tokenize cleaned text into integer sequences
   - Pad sequences to a fixed length
   - Build an `Embedding → LSTM → Dropout → Dense(softmax)` model
   - Train and evaluate the sequence model on the same sentiment task

## Results
- The notebook includes sample predictions in `results/sample_predictions.txt`.
- Both baseline models and the sequence model achieved strong performance in the current evaluation setup.
- Stored notebook outputs show:
  - Logistic Regression accuracy: `1.0000`
  - Multinomial Naive Bayes accuracy: `1.0000`
  - LSTM Test Accuracy: `1.0000`

## Observations
- The dataset appears well-separated for the sentiment classes used in this assignment.
- Baseline text classifiers using TF-IDF and Bag-of-Words are effective for this problem.
- The LSTM model provides a sequence-based alternative and is useful when text order and context matter.
- Perfect accuracy may indicate a small or very clean dataset, so caution is needed when generalizing to new data.
- Additional validation with external data or cross-validation would help confirm model robustness.

## How to use
1. Open `notebook.ipynb` in Jupyter or VS Code.
2. Install the project dependencies.
3. Run the notebook cells in order to reproduce the data analysis, model training, and evaluation.

## Notes
- The LSTM section requires TensorFlow/Keras. If TensorFlow is unavailable, the notebook still explains the architecture and design.
- The `requirement.txt` file is currently empty, so install dependencies manually if needed.
