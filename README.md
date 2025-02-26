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

#### API Usage  
**Endpoint:**  
```http
POST https://housing-in-melbourne.onrender.com/predict








# Housing In Melbourne: Beyond the Price Tag

## 📌 Project Overview  
In this project, I developed a **machine learning model** to predict **house prices in Melbourne, Australia**. The model takes in **suburb, building area, latitude, and longitude** to generate property price estimates in USD.  

To enhance predictive accuracy, I built **three sub-models** focusing on different price factors:
1. **House Price by Building Area** – Predicts price based on property size.
2. **House Price by Location** – Examines price variations using latitude/longitude.
3. **House Price by Suburb** – Evaluates pricing trends across different suburbs.
4. **Beyond the Price Tag** – A final integrated model combining all three factors.

The project is **deployed in two ways**:
- **Jupyter Widgets** – An interactive price predictor within the notebook.
- **Flask API** – A web-based prediction service.

---

## 🗂 Dataset & Processing  

The dataset was sourced from **Domain.com.au**, containing property transaction records including:  
- **Suburb, building area, latitude, longitude, price, and more**  
- **Price converted to USD (2018 rates)**  
- **One-Hot Encoding** for categorical features like **suburb**  

### 🏗 Data Preprocessing with `DataProcessor` and `DataTransformer`
To ensure the dataset was **clean and optimized for modeling**, I modularized the data wrangling process using two key classes:

- **`DataProcessor` (wrangle function-turned module)**:  
  - Initial preprocessing, handling missing values, and renaming columns (`Lattitude → Latitude`, `Longtitude → Longitude`).
  - Filtering **outliers** using IQR-based removal.
  - Saving the cleaned dataset for further transformations.

- **`DataTransformer` (updated_wrangle function-turned module)**:  
  - Converts categorical variables (e.g., **Suburb**) into **one-hot encoded features**.
  - Caps extreme values instead of outright removal for **BuildingArea** and **Price_USD**.
  - Drops irrelevant features and applies final transformations for training.

By structuring the **data pipeline** using these two classes, I ensured **consistency and reusability** across different model versions.

---

## 🔬 Exploratory Data Analysis (EDA)  

### Key Insights:
- **Price Trends**  
  - Properties within **10 km of the CBD** have the highest prices.  
  - **Central & Eastern suburbs** are the most expensive.  
  - **Western & Northern suburbs** offer more affordable options.  

- **Building Area vs. Price**  
  - Larger properties generally cost more, but **the relationship isn’t linear**.  
  - **Smaller properties (<150 sqm)** show greater price variation.  

- **Suburb Influence**  
  - Some suburbs show **significantly higher prices** for similar-sized properties.  

Detailed EDA visualizations (scatter plots, maps, and histograms) are included in the reports.

---

## 🔧 Model Development & Performance  

### **1️⃣ House Price by Building Area**  
- **Baseline MAE**: $241,499.33  
- **Testing MAE**: $233,855.27  
- **Key Finding**: Building area **alone is not enough** to predict prices accurately.  

### **2️⃣ House Price by Location**  
- **Baseline MAE**: $241,499.33  
- **Testing MAE**: $217,255.91  
- **Key Findings**:  
  - **Central & Eastern suburbs** have higher prices.  
  - **Western & Northern** suburbs are more budget-friendly.  

### **3️⃣ House Price by Suburb**  
- **Model Used**: Random Forest  
- **Findings**:  
  - Model performs well but **struggles with high-priced properties**.  
  - Adding **location & building area** improves accuracy.  

### **4️⃣ Beyond the Price Tag (Final Model)**  
- **Model Used**: Ridge Regression  
- **Performance Metrics**:  
  - **Baseline MAE**: $276,482.90  
  - **Training MAE**: $157,843.17  
  - **Testing MAE**: $159,148.28  
- **Key Finding**: Combining **suburb, location, and size** **significantly improves predictions**.

---

## 🚀 Model Deployment  

### **1️⃣ Interactive Jupyter Widget (Google Colab)**  
To use the interactive price prediction widget, open the notebook on **Google Colab**:  

### 🎛 Try the Interactive Widget on Google Colab
Click below to open:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NdubuakuMiracle/housing-in-melbourne/blob/main/reports/app/deployed_interactive_widget.ipynb)

### 🚀 How to Use:
1️⃣ **Click the "Open in Colab" button** above.  
2️⃣ **Run all cells** to load `X_test.csv` & `model.pkl` directly from GitHub.  
3️⃣ **Use the interactive widget to predict house prices!**  

---

### **2️⃣ Flask API (Deployed on Render)**  
I also deployed a **Flask web app** for making predictions via API.

### 🌍 Try the Live Flask API
Click below to test the Flask web app:

[![Open Flask App](https://img.shields.io/badge/Open%20Flask%20App-Click%20Here-brightgreen)](https://housing-in-melbourne.onrender.com)

### 📡 API Usage  
**Endpoint:**  
`POST https://housing-in-melbourne.onrender.com/predict`

#### **Example Request:**
```json
{
  "Suburb": "Richmond",
  "BuildingArea": 150,
  "Latitude": -37.81,
  "Longitude": 144.99
}
