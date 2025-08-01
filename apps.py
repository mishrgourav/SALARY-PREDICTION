import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Salary Prediction App")
st.title("💼 Data Science Salary Prediction")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV only)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    if 'salary_in_usd' not in df.columns:
        st.error("The dataset must have a column named 'salary_in_usd'.")
    else:
        X = df.drop('salary_in_usd', axis=1)
        y = df['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models
        rf = RandomForestRegressor()
        gb = GradientBoostingRegressor()
        voting = VotingRegressor(estimators=[('rf', rf), ('gb', gb)])

        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        voting.fit(X_train, y_train)

        def evaluate(model, name):
            y_pred = model.predict(X_test)
            st.subheader(f"{name} Results")
            st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"**MSE**: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            st.write(f"**R² Score**: {r2_score(y_test, y_pred):.4f}")

        evaluate(rf, "🌲 Random Forest")
        evaluate(gb, "🌱 Gradient Boosting")
        evaluate(voting, "🧠 Voting Regressor")

        # Feature Importance (Random Forest)
        st.subheader("🔍 Feature Importances (Random Forest)")
        importances = rf.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Relative Importance")
        st.pyplot(fig)
