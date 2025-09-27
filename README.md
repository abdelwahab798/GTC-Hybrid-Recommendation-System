# 📌 Hybrid Recommendation System  

This project implements a **Hybrid Recommendation System** that combines **Collaborative Filtering (SVD)** with **Content-Based Filtering** (using textual data ). This system helps users discover relevant products by combining collaborative signals from ratings with semantic similarity from product reviews. 


---
# 🚀 Project Workflow  


### 1. Data Preprocessing  
- ✅ Handled missing values.  
- ✅ Verified no duplicate entries.  
- ✅ Fixed data types.  

### 2. Feature Engineering  
- Extracted time-based features: `year`, `month`, `day`, `dayofweek`.  
- Created new features:  
  - `helpfulness_ratio = HelpfulnessNumerator / HelpfulnessDenominator`  
  - `time_weight = 1 / (1 + (year - df["year"]))`  
  - `final_score`, `final_score2`, `final_score3` (weighted variations of score).  

### 3. Exploratory Data Analysis (EDA)  
- 📈 Reviews increased significantly over the years.  
- ⭐ 60% of ratings are **5**, showing a strong positive skew.  
- 👥 4 users are clear outliers with much higher review counts.  
- 📦 One product has a significantly higher number of reviews.  
- 📅 Ratings dip mid-week and rise on weekends.  
- 🏆 **1999–2000** best years for ratings, **2001** worst.  
- 📊 Post-2012: stable but fluctuating ratings trend.  

---

## 🧩 Models  

### 1. Collaborative Filtering (CF)  
- Algorithm: **SVD** (after hyperparameter tuning).  
- Best hyperparameters:  
  ```python
  {'n_factors': 50, 'n_epochs': 80, 'lr_all': 0.02, 'reg_all': 0.08}
  ```
- Features used: `(UserID, ProductID, final_score2)`  


---

### 2. Content-Based Filtering  

#### A. Text + Summary Model  
- Combined `Text` + `Summary` → `text_all`.  
- Applied **TF-IDF** vectorization.  
- Built **Nearest Neighbors** system for product similarity.  

Workflow:  
1. User enters `User ID` & `Product ID`.  
2. System calculates α (based on user activity).  
3. Retrieves similar products (text-based).  
4. Combines with SVD scores.  
5. Displays top recommendations interactively.  


---

### 3. Hybrid Models  
- **Hybrid **: (Text + Summary + CF)  
  - Weighted Hybrid (α depends on user’s number of ratings).
  - Final Score = α * Collaborative Score + (1 - α) * Content Score  
   
---

## 🔎 Results

### Collaborative Filtering
- RMSE = **0.9334**  
- MAE  = **0.8112**

### Content-Based Filtering
- Precision@5 = **0.40**  
- Recall@5    = **0.67**  
- NDCG@5      = **0.65**

### Hybrid (CF + Content)
- Dynamic α depending on user activity:  
  - New users (α=0.3 → More Content-based)  
  - Medium users (α=0.5)  
  - Power users (α=0.7 → More Collaborative)  


## 🖥️ Deployment  
- Built an **interactive interface** to display recommendations.  

---

## ⚙️ Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn, Surprise, NLTK)  
- **NLP**: TF-IDF, text preprocessing  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Streamlit  

---

## 📂 Project Structure  
```
├── notebooks/           # Jupyter notebooks (EDA, Cleaning, Feature Engineering)
├── models/              # Trained models (SVD, Hybrid)
├── app/                 # Deployment files (Streamlit, Chatbot)
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

---

## ▶️ How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/abdelwahab798/GTC-Hybrid-Recommendation-System.git
   cd hybrid-recommender
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:  
   ```bash
   streamlit run app/app_3.py
   ```

---
## 📂 Project Resources

- **Dataset:** [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)  
- **Collaborative Filtering Model:** [Download from Google Drive](https://drive.google.com/file/d/1Mbv1LRy1gxP5jzH8zDCbUML7JBn3JVzn/view?usp=sharing)



