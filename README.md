# 🎬 Netflix Content-Based Recommendation System

## 📌 Overview

This project builds a content-based recommendation system using Netflix title metadata. 
The system recommends similar movies and TV shows based on textual features such as genre, description, cast, director, and country.

Unlike collaborative filtering methods, this system does not rely on user behavior data, making it suitable for cold-start scenarios.

---

## 🚀 Features

- Content-based recommendation using TF-IDF
- Cosine similarity for similarity computation
- Top-N recommendation generation
- Interactive web application using Streamlit
- Card-style recommendation display for improved user experience

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, cosine similarity)
- Streamlit (Web Application)
- Matplotlib (Exploratory Data Analysis)

---

## 📊 Dataset

- Source: Kaggle Netflix Dataset
- Size: ~8800 titles
- Key Features:
  - Title
  - Genre (listed_in)
  - Description
  - Cast
  - Director
  - Country
  - Release Year

---

## ⚙️ Methodology

1. Data Cleaning  
   - Handled missing values  
   - Converted date features  
   - Removed duplicates  

2. Feature Engineering  
   - Combined genre, description, cast, director, and country into a single text feature  

3. Text Representation  
   - Applied TF-IDF vectorization to transform text into numerical features  

4. Similarity Computation  
   - Used cosine similarity to measure similarity between titles  

5. Recommendation System  
   - Generated Top-N similar titles based on user input  

---

## 🎥 Demo

Run the application locally:

```bash
streamlit run app.py
