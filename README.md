# 🔍 Duplicate Question Pair Detector (NLP Backend & Streamlit Web App)

An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline designed to detect semantic similarity between question pairs. The system extracts advanced token, length, and fuzzy string matching features to classify whether two text inquiries share the exact same underlying intent, deployed as an interactive web interface.

---

## 📈 Performance Summary
*   **Best Performing Model**: CatBoost Classifier
*   **Validation Accuracy**: **79.70%** (Outperformed baseline Logistic Regression and Linear SVM architectures)
*   **Feature Dimension Vector**: 6,022 hybrid features (22 engineered metrics + 6,000 sparse BoW text vectors)

---

## 📦 Project Artifacts & Data Source
Due to GitHub's file size limitations, the large training dataset is hosted externally on Google Drive:

*   **Dataset Download**: [Download train.csv via Google Drive](https://drive.google.com/file/d/1-gEU2jqPGtyX4-rUS8N9yhHhQDZJj2UO/view?usp=sharing)

> **Note**: To run this pipeline locally, download `train.csv` from the link above and place it directly into the root directory of this repository.

---

## 📂 Repository Structure
```text
├── bow-with-preprocessing-and-advanced-features.ipynb       # Core Jupyter Notebook (EDA, Feature Engineering, Training)
├── app.py               # Interactive Streamlit Web Interface Script
├── model.pkl            # Pre-trained CatBoost Classifier binary weights
├── cv.pkl               # Fitted CountVectorizer vocabulary token mapping
├── .gitignore           # Excludes train.csv and cache runtimes from tracking
└── requirements.txt     # Python production library configurations
```

---

## 📐 Engineered Features (22 Dimensions)
The project bypasses basic keyword matching traps by calculating **22 structural metrics**:
1.  **Basic Metrics**: Character length, word split count, and total common words share ratio.
2.  **Advanced Token Metrics**: Continuous interaction over non-stopwords (`cwc`), stopwords (`csc`), and tokens (`ctc`) tracking both minimum and maximum intersection boundaries.
3.  **Positional Tracking**: Exact structural flags tracking whether the first or last tokens match.
4.  **Substring Constraints**: Longest common substring ratios via distance heuristics.
5.  **Fuzzy Math Similarities**: Advanced partial string alignment, sort tokens, and sequence set intersections via the `FuzzyWuzzy` matching engine.

---

## 🚀 Local Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up Your Python Environment & Install Dependencies
Ensure you are using Python 3.9+ and run:
```bash
pip install -r requirements.txt
```

### 3. Fetch the Dataset
Download `train.csv` from the [Google Drive Link](https://drive.google.com/file/d/1-gEU2jqPGtyX4-rUS8N9yhHhQDZJj2UO/view?usp=sharing) and save it inside your repository root folder.

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Your default browser will instantly spin up a local development page hosting address at **`http://localhost:8501`**.

---

## 🔮 Usage Examples
Inside the web app dashboard, you can test the pipeline using these inputs:
*   **Duplicate Pair Example**:
    *   *Question 1*: `How can I learn machine learning from scratch?`
    *   *Question 2*: `What is the best way to learn machine learning from scratch?`
    *   *Verdict*: **🚨 DUPLICATE DETECTED**
*   **Keyword Trap Example**:
    *   *Question 1*: `How does a computer virus work?`
    *   *Question 2*: `What are the biological symptoms of a flu virus?`
    *   *Verdict*: **✅ UNIQUE QUESTIONS**
