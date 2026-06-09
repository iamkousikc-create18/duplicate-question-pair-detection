import streamlit as st
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix
from fuzzywuzzy import fuzz
import distance
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
try:
    STOP_WORDS = stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = stopwords.words("english")

# --- Step 1: Page Configuration & Asset Loading ---
st.set_page_config(page_title="Duplicate Question Detector", page_icon="🔍", layout="centered")

@st.cache_resource
def load_assets():
    # Load your saved models and vectorizers
    cat_model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('cv.pkl', 'rb'))
    return cat_model, cv

try:
    cat_model, cv = load_assets()
except FileNotFoundError:
    st.error("🚨 Configuration Error: 'model.pkl' or 'cv.pkl' not found in this directory. Please verify file placement.")
    st.stop()

# --- Step 2: Exact Instructor Preprocessing Engine ---
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    
    # Text decontraction dictionary from page 2-4 of project file
    q = q.replace("i've", "i have").replace("wasn't", "was not").replace("n't", " not").replace("'re", " are").replace("'ll", " will")
    
    # Strip HTML and special characters
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(re.compile(r'\W'), ' ', q).strip()
    return q

# --- Step 3: Exact 22-Feature Extractor Logic ---
def extract_query_features(q1, q2):
    SAFE_DIV = 0.0001
    
    q1_cleaned = preprocess(q1)
    q2_cleaned = preprocess(q2)
    
    q1_tokens = q1_cleaned.split()
    q2_tokens = q2_cleaned.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return None, None, None
        
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    cwc_min = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    cwc_max = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    last_word_eq = float(int(q1_tokens[-1] == q2_tokens[-1]))
    first_word_eq = float(int(q1_tokens[0] == q2_tokens[0]))
    
    abs_len_diff = float(abs(len(q1_tokens) - len(q2_tokens)))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    
    strs = list(distance.lcsubstrings(q1_cleaned, q2_cleaned))
    longest_substr_ratio = 0.0 if len(strs) == 0 else len(strs[0]) / (min(len(q1_cleaned), len(q2_cleaned)) + 1)
    
    fuzz_ratio = fuzz.QRatio(q1_cleaned, q2_cleaned)
    fuzz_partial_ratio = fuzz.partial_ratio(q1_cleaned, q2_cleaned)
    token_sort_ratio = fuzz.token_sort_ratio(q1_cleaned, q2_cleaned)
    token_set_ratio = fuzz.token_set_ratio(q1_cleaned, q2_cleaned)
    
    # 22 numeric feature configuration mapping training parameters
    numeric_features = [
        len(q1_cleaned), len(q2_cleaned), len(q1_tokens), len(q2_tokens),
        common_word_count, (len(q1_tokens) + len(q2_tokens)), 
        (common_word_count / (len(q1_tokens) + len(q2_tokens) + SAFE_DIV)),
        cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, 
        last_word_eq, first_word_eq, abs_len_diff, mean_len, longest_substr_ratio,
        fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio
    ]
    
    return numeric_features, q1_cleaned, q2_cleaned

# --- Step 4: Streamlit Web UI Layout ---
st.title("🔍 NLP Duplicate Question Pair Detector")
st.write("Provide two questions below to cross-examine their semantic intent via our optimized **CatBoost** backend pipeline.")

# Text inputs for questions
question1 = st.text_input("First Question:", placeholder="e.g., How do I learn Python?")
question2 = st.text_input("Second Question:", placeholder="e.g., What is the best way to study Python?")

if st.button("Run Duplicate Examination", type="primary"):
    if not question1.strip() or not question2.strip():
        st.warning("⚠️ Action required: Both question fields must be populated.")
    else:
        with st.spinner("Processing text features and running inference..."):
            numeric_features, q1_cl, q2_cl = extract_query_features(question1, question2)
            
            if numeric_features is None:
                st.success("Verdict: **UNIQUE QUESTIONS** (Empty query parameters handled)")
            else:
                # Text structural vectorization
                q1_cv = cv.transform([q1_cl])
                q2_cv = cv.transform([q2_cl])
                text_features = hstack([q1_cv, q2_cv])
                
                # Stack dense configurations and sparse matrices 
                X_query = hstack([csr_matrix([numeric_features]), text_features])
                
                # Execute pipeline inference
                prediction = cat_model.predict(X_query)
                probabilities = cat_model.predict_proba(X_query)[0]
                
                # Display beautifully stylized UI response panels
                st.subheader("📋 Classification Verdict")
                if prediction == 1:
                    confidence = probabilities[1] * 100
                    st.error(f"🚨 **DUPLICATE DETECTED** (Confidence Summary Score: {confidence:.2f}%)")
                    st.info("Both statements point to the exact same core request or semantic inquiry.")
                else:
                    confidence = probabilities[0] * 100
                    st.success(f"✅ **UNIQUE QUESTIONS** (Confidence Summary Score: {confidence:.2f}%)")
                    st.info("These inquiries are distinct, tracking separate objectives or independent topics.")
