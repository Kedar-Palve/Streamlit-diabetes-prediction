import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

st.write("""
    # Simple Diabetes Prediction App

    ### This app predicts the diabetes in pregnant women!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Glucose = st.sidebar.slider('Glucose', 0.0, 199.0, 18.0)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0.0, 122.0, 10.0)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 80.3)
    Age = st.sidebar.slider('Age', 21.0, 81.0, 5.5)
    data = {'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'Insulin': Insulin,
            'Age': Age
    }

    features = pd.DataFrame(data, index=[0])
    return features

f = user_input_features()

st.subheader('User Input Parameters')
st.write(f)

warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Use the model to predict the outcomes for the test set
test_preds = clf.predict(X_test)

# Save the predicted outcomes to a CSV file
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds})
pred_df.to_csv('predictions.csv', index=False)

st.write(pred_df)