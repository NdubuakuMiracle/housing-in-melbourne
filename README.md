# Housing in Melbourne: Beyond the Price Tag

## Project Overview  
This project predicts house prices in Melbourne, Australia, using machine learning models trained on **suburb, building area, latitude, and longitude**.  

### Key Features
- Developed multiple models for prediction:  
  - **House Price by Building Area** – Estimates price based on property size.  
  - **House Price by Location** – Examines price variations using latitude/longitude.  
  - **House Price by Suburb** – Analyzes pricing trends across different suburbs.  
  - **Beyond the Price Tag** – A final **integrated model combining all three factors** for improved accuracy.  
- **Deployment Options:**  
  - **Jupyter Widget** – Interactive price predictor inside a notebook.  
  - **Flask API** – Web-based prediction service hosted on Render.  

---

## Dataset & Processing  

### Data Source  
- Property transaction data from **Domain.com.au**  
- **Key Features:** suburb, building area, latitude, longitude, price  
- **Price Conversion:** AUD to USD (2018 rates)  
- **Categorical Features:** One-hot encoding for **suburb**  

### Data Processing Pipeline
The data wrangling process was modularized into two key components:  

#### `DataProcessor`
- Handles raw data cleaning, renames columns, and removes missing values.  
- Filters outliers using **IQR-based removal**.  

#### `DataTransformer`
- Converts categorical variables (**Suburb**) into **one-hot encoded features**.  
- Caps extreme values instead of outright removal for **BuildingArea & Price_USD**.  
- Drops irrelevant features and applies final transformations for training.  

This modular approach ensures **consistency and reusability** across different models.  

---

## Exploratory Data Analysis (EDA)  

### Key Findings
- **Price Trends:**  
  - Properties within **10 km of the CBD** have the highest prices.  
  - **Central & Eastern suburbs** are more expensive, while **Western & Northern suburbs** are more affordable.  
- **Building Area vs. Price:**  
  - Larger properties **generally cost more**, but the relationship **isn’t linear**.  
  - Smaller properties (<150 sqm) **show greater price variation**.  
- **Suburb Influence:**  
  - Some suburbs **command significantly higher prices** for similar-sized properties.  

*Visualizations (scatter plots, maps, histograms) are included in the report.*  

---

## Model Development & Performance  

### 1️⃣ House Price by Building Area  
- **Baseline MAE:** $241,499.33  
- **Testing MAE:** $233,855.27  
- **Key Finding:** Building area **alone is insufficient** for accurate predictions.  

### 2️⃣ House Price by Location  
- **Baseline MAE:** $241,499.33  
- **Testing MAE:** $217,255.91  
- **Key Finding:** **Location significantly impacts pricing trends.**  

### 3️⃣ House Price by Suburb  
- **Model Used:** Random Forest  
- **Findings:**  
  - Model performs well but **struggles with high-priced properties**.  
  - Adding **location & building area** improves accuracy.  

### 4️⃣ Beyond the Price Tag (Final Model)  
- **Model Used:** Ridge Regression  
- **Performance Metrics:**  
  - **Baseline MAE:** $276,482.90  
  - **Training MAE:** $157,843.17  
  - **Testing MAE:** $159,148.28  
- **Key Finding:** Combining **suburb, location, and size** significantly improves predictions.  

---

## 🚀 Model Deployment  

### 1️⃣ Interactive Jupyter Widget (Google Colab)  
This project features an **interactive price prediction widget** inside a Jupyter Notebook, allowing users to test predictions dynamically.  

### Try the Interactive Widget on Google Colab  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NdubuakuMiracle/housing-in-melbourne/blob/main/reports/app/deployed_interactive_widget.ipynb)  

#### How to Use:  
1. **Click the "Open in Colab" button** above.  
2. **Run all cells** (Loads `X_test.csv` & `model.pkl` directly from GitHub).  
3. **Use the interactive widget to predict house prices.**  

---

### 2️⃣ Flask API (Deployed on Render)  
A **Flask API** is deployed on Render for real-time predictions.  

### Try the Live Flask API  
[![Open Flask App](https://img.shields.io/badge/Open%20Flask%20App-Click%20Here-brightgreen)](https://housing-in-melbourne.onrender.com)

### 📡 API Usage  
**Endpoint:**  
POST https://housing-in-melbourne.onrender.com/predict

#### **Example Request:**
json
{
  "Suburb": "Richmond",
  "BuildingArea": 150,
  "Latitude": -37.81,
  "Longitude": 144.99
}
