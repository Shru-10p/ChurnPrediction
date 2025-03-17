import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, render_template


# Load and preprocess the dataset
def load_and_preprocess_data():
    # Replace 'customer_churn_data.csv' with your actual dataset path
    df = pd.read_csv('customer_churn_data.csv')

    # Encode categorical variables
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = le.fit_transform(df[column])

    # Drop unnecessary columns
    df.drop('customerID', axis=1, inplace=True)

    # Split the data into features and target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

# Train the Random Forest model
def train_modelRandomForest(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [RandomForestClassifier(n_estimators=100, random_state=42), LogisticRegression(), KNeighborsClassifier(), SVC()]
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Model accuracy: {}'.format(accuracy))
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

    # Save the model to disk
    joblib.dump(model, 'random_forest_model.pkl')



# Create Flask app
app = Flask(__name__)

# Load and preprocess data
X, y = load_and_preprocess_data()

# Train the model
train_modelRandomForest(X, y)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text='Churn Prediction: {}'.format(prediction[0]))

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
