# American Express User Exit Prediction  
### Customer Churn Prediction Using Artificial Neural Networks (ANN)

---

## 🚀 Overview
This project predicts **customer churn (user exit)** for **American Express** using a supervised machine-learning model built with an **Artificial Neural Network (ANN)**.  
The goal is to identify customers who are likely to leave so the company can take proactive retention actions.

---

## 🎯 Objectives
- Study customer behavior and churn patterns  
- Clean and preprocess financial customer data  
- Build and train an ANN model using TensorFlow/Keras  
- Evaluate the model using standard classification metrics  
- Provide insights for business decision-making  

---

## 📂 Dataset Description
The dataset contains customer information including:

- Demographics  
- Spending and transaction behavior  
- Account activity  
- Payment history  
- Credit utilization  
- Customer support interactions  

**Target Column:** `Exited`  
- `1` → Customer exited  
- `0` → Customer retained  

> ⚠️ *The dataset used is academic/synthetic and does not include real American Express proprietary data.*

---

## 🧹 Data Preprocessing
Key preprocessing steps:

- Handling missing values  
- Encoding categorical variables (LabelEncoder / One-Hot Encoding)  
- Scaling numerical features (StandardScaler)  
- Splitting into training & testing datasets  
- Optional feature engineering based on insights  

---

## 🧠 ANN Model Architecture
The model was built using **TensorFlow/Keras**.
- Input Layer
- ↓
- Dense Layer (ReLU) → Dropout
- ↓
- Dense Layer (ReLU) → Dropout
- ↓
- Output Layer (Sigmoid)

---

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Metrics:** Accuracy, Precision, Recall, F1-Score  

---

## 📈 Final Results (Based on Your Dataset)

| Metric | Value |
|--------|--------|
| **Accuracy** | **85.59%** |
| **Precision** | **75.47%** |
| **Recall** | **47.50%** |
| **F1 Score** | **58.31%** |

You can include your confusion matrix or training graphs as images if available.

---

## 📊 Visualizations
The project includes:

- Correlation Matrix  
- Feature Distribution Plots  
- Model Training Curves (Loss & Accuracy)  
- Confusion Matrix  

---

## 🛠 Tech Stack
- Python 3.x  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook / Google Colab  

---

## 💡 Business Insights

- Customers with **irregular payment behaviors** show higher churn risk.  
- **Low engagement** (transactions or support calls) strongly correlates with exits.  
- High-risk customers can be targeted with **personalized retention strategies** to reduce churn.

---

## 📄 License

This project is developed for **educational and research purposes only**.

---

## 🙋‍♂️ Author

**Chawhan Bhoomika**  
BCA Final Year Project  
**Project:** American Express User Exit Prediction



