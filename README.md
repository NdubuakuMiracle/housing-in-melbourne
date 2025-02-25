# Beyond the Price Tag: Predicting Melbourne House Prices

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/beyond_the_price_tag_notebook.ipynb)  

#### **Steps to Use on Colab:**  
1. Open the notebook.  
2. Upload `X_train.csv` and `price_by_sub_area_lat_lon_model.pkl`.  
3. Run all cells to load the model.  
4. Use the **interactive widget** to predict house prices.

---

### **2️⃣ Flask API**  
I also deployed a Flask web app for making predictions via API.

#### **How to Run Flask Locally**  
```sh
python app.py
