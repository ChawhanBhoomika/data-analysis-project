**American Express User Exit Prediction**
-
Customer Churn Prediction Using Artificial Neural Networks (ANN)
--
🚀 **Overview**
---

This project predicts customer churn (user exit) for American Express using a supervised machine-learning model built with an Artificial Neural Network (ANN).
The goal is to identify customers who are likely to leave, enabling proactive retention strategies.

🎯 **Objectives**
------------

Study customer behavior and churn patterns
Clean and preprocess structured financial customer data
Build and train an ANN using TensorFlow/Keras
Evaluate model performance using classification metrics
Provide insights to support business decisions

##📂 **Dataset Description**

-The dataset contains customer information including:
-Demographics
-Spending & transaction behavior
-Account activity
-Payment history
-Credit utilization
-Customer support interactions
###Target column: Exited
1 → Customer exited
0 → Customer retained
⚠️ Dataset used is academic/synthetic and does not include real American Express proprietary data.

##🧹 **Data Preprocessing**

###Key preprocessing steps include:
-Handling missing values
-Encoding categorical variables (LabelEncoder / One-Hot Encoding)
-Scaling numerical features (StandardScaler)
-Splitting into train/test sets
-Feature engineering (optional based on model insights)

##🧠**Model Architecture — ANN**

-Built using TensorFlow/Keras:
-Input Layer  →  Dense Layer (ReLU) → Dropout  
              →  Dense Layer (ReLU) → Dropout  
              →  Output Layer (Sigmoid)

###Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy, Precision, Recall, F1-Score

📊 **Visualizations**

The project includes:
-
Correlation matrix

Feature distribution plots

Model training curves (loss & accuracy)

Confusion matrix

🛠 Tech Stack

Python 3.x

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

Jupyter Notebook / Google Colab

📦 Installation & Usage
1. Clone the repository
git clone https://github.com/your-username/american-express-user-exit.git
cd american-express-user-exit

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook
jupyter notebook


Then open:
American_Express_User_Exit.ipynb

📝 Folder Structure (Recommended)
📁 american-express-user-exit
 ┣ 📂 data
 ┃ ┗ dataset.csv
 ┣ 📂 notebooks
 ┃ ┗ American_Express_User_Exit.ipynb
 ┣ 📂 models
 ┃ ┗ ann_model.h5
 ┣ 📄 README.md
 ┣ 📄 requirements.txt

💡 Business Insights

Customers with irregular payment patterns show higher churn probability.

Low engagement (transactions/support calls) correlates with higher exit risk.

High-risk customers can be targeted with personalized retention strategies.

📄 License

This project is intended for educational and research purposes only.

🙋‍♂️ Author

Your Name
BCA Final Year Project
Project: American Express User Exit Prediction
