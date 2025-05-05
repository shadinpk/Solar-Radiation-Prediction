import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Solar Radiation Prediction", layout="centered")
st.title("Solar Radiation Prediction using XGBoost")
st.write("Upload your own data or explore model performance below.")


try:
    df= pd.read_csv("SolarPrediction.csv")
    st.success(" Dataset loaded successfully.")
    st.dataframe(df.head())
except Exception as e:
    st.error(f" Failed to load default dataset: {e}")
    st.stop()


df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df['Hour'] = df['Datetime'].dt.hour
df['Minute'] = df['Datetime'].dt.minute
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Weekday'] = df['Datetime'].dt.weekday


df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)


df.drop(columns=['UNIXTime', 'Data', 'Time', 'Datetime', 'TimeSunRise', 'TimeSunSet', 'Hour', 'Month'], inplace=True)


X = df.drop(columns=['Radiation'])
y = df['Radiation']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


st.subheader(" Model Performance")
st.write(f"**Mean Squared Error:** {round(mse, 2)}")
st.write(f"**R² Score:** {round(r2, 4)}")      


st.subheader("Feature Importance")
fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax)
st.pyplot(fig)


st.subheader(" Upload Your Own CSV to Predict Radiation")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        input_df['Datetime'] = pd.to_datetime(input_df['Data'] + ' ' + input_df['Time'])
        input_df['Hour'] = input_df['Datetime'].dt.hour
        input_df['Minute'] = input_df['Datetime'].dt.minute
        input_df['Month'] = input_df['Datetime'].dt.month
        input_df['Day'] = input_df['Datetime'].dt.day
        input_df['Weekday'] = input_df['Datetime'].dt.weekday
        input_df['Hour_sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
        input_df['Hour_cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)
        input_df['Month_sin'] = np.sin(2 * np.pi * input_df['Month'] / 12)
        input_df['Month_cos'] = np.cos(2 * np.pi * input_df['Month'] / 12)

        input_df.drop(columns=['UNIXTime', 'Data', 'Time', 'Datetime', 'TimeSunRise', 'TimeSunSet', 'Hour', 'Month'], inplace=True)

        predictions = model.predict(input_df)
        input_df['Predicted Radiation'] = predictions

        st.success(" Prediction completed!")
        st.dataframe(input_df[['Minute', 'Day', 'Weekday', 'Predicted Radiation']].head())

       
        csv_download = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Predictions as CSV", data=csv_download, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f" Error processing file: {e}")


st.write(" App loaded successfully!")
# Solar-Radiation-Prediction
# Solar-Radiation-Prediction
