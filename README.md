# Bank Churn Prediction using Artificial Neural Network (ANN)

## Project Overview
This project predicts whether a bank customer is likely to leave the bank using an Artificial Neural Network (ANN). The model uses customer information to identify potential churn, helping banks understand customer behavior and improve customer retention.

## Objectives
- Predict customer churn using Deep Learning.
- Perform data preprocessing and categorical encoding.
- Scale numerical features for better model performance.
- Build and train an ANN classification model.
- Deploy an interactive web application using Streamlit.

## Technologies Used
- **Python** – Programming language
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Data preprocessing and feature scaling
- **TensorFlow / Keras** – ANN model development
- **Streamlit** – Web application and deployment
- **Pickle** – Saving preprocessing objects

## Project Workflow
1. Data collection and exploration
2. Data preprocessing and feature engineering
3. Categorical feature encoding
4. Feature scaling using StandardScaler
5. ANN model building and training
6. Model evaluation
7. Saving the trained model and preprocessing objects
8. Deploying the application using Streamlit

## Project Structure

```text
Bank_churn_prediction_ANN/
├── app.py
├── model.h5
├── StandardScaler.pkl
├── OneHotEncoding.pkl
├── LabelEncoder.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/nithenath/Bank_churn_prediction_ANN.git
cd Bank_churn_prediction_ANN
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

The application will open in your browser, allowing you to interact with the churn prediction model.

## Model Description
The Artificial Neural Network learns patterns from customer data to classify customers according to their likelihood of leaving the bank. The trained model and preprocessing objects are saved and reused by the Streamlit application to process user inputs consistently.

## Applications
- Customer retention analysis
- Identification of customers at risk of churn
- Data-driven banking decisions
- Customer relationship management

## Future Improvements
- Improve prediction performance through hyperparameter tuning.
- Add model evaluation metrics and visualizations.
- Explore additional machine learning and deep learning models.
- Enhance the user interface and deployment workflow.

## Author
**Nithenath Vairavan**

GitHub: [nithenath](https://github.com/nithenath)

---

*This project was developed for educational purposes to explore Artificial Neural Networks and their application to customer churn prediction.*
