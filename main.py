import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template


def load_and_preprocess_data():
    df = pd.read_csv('customer_churn_data.csv')

    le = LabelEncoder()
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            continue
        df[column] = le.fit_transform(df[column])

    df.drop('customerID', axis=1, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

def train_modelRandomForest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_modelLogisticRegression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_modelKNN(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

app = Flask(__name__)

X, y = load_and_preprocess_data()

model_rf = train_modelRandomForest(X, y)
model_lr = train_modelLogisticRegression(X, y)
model_knn = train_modelKNN(X, y)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=="GET":
        return render_template('index.html')
    features = [float(x) for x in request.form.values()]
    prediction_rf = model_rf[0].predict([features])
    prediction_lr = model_lr[0].predict([features])
    prediction_knn = model_knn[0].predict([features])
    total_acc = model_knn[1] + model_lr[1] + model_rf[1]
    avg_pred = (model_knn[1]*prediction_rf+model_lr[1]*prediction_lr+model_rf[1]*prediction_knn)/(total_acc)
    return render_template('index.html',
                            prediction_rf=prediction_rf[0],
                            acc_rf = model_rf[1],
                            prediction_lr=prediction_lr[0], 
                            acc_lr = model_lr[1],
                            prediction_knn=prediction_knn[0], 
                            acc_knn = model_knn[1],
                            avg_pred = round(avg_pred[0]))

if __name__ == "__main__":
    app.run(debug=False)
