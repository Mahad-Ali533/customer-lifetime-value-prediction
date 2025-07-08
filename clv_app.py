import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# === Title ===
st.title("📈 Customer Lifetime Value (CLV) Prediction")

# === Upload CSV File ===
st.markdown("Upload a CSV file with **Recency**, **Frequency**, **Monetary**, and **CLV** columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = {'Recency', 'Frequency', 'Monetary', 'CLV'}
        if not required_columns.issubset(df.columns):
            st.error("❌ CSV must contain the following columns: Recency, Frequency, Monetary, CLV")
        else:
            st.success("✅ Data loaded successfully!")
            st.write("Sample data:", df.head())

            # Split Data
            X = df[['Recency', 'Frequency', 'Monetary']]
            y = df['CLV']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("🔍 Model Evaluation")
            st.write(f"**Mean Squared Error:** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.4f}")

            # Show predictions
            pred_df = pd.DataFrame({
                'Actual CLV': y_test.values,
                'Predicted CLV': y_pred
            })
            st.subheader("📊 Prediction Results")
            st.dataframe(pred_df.head())

            # Download Excel Option
            def to_excel(pred_df, mse, r2):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    pred_df.to_excel(writer, index=False, sheet_name='Predictions')
                    pd.DataFrame({'Metric': ['MSE', 'R2'], 'Value': [mse, r2]}).to_excel(writer, index=False, sheet_name='Summary')
                return output.getvalue()

            st.download_button(
                label="📥 Download Results (Excel)",
                data=to_excel(pred_df, mse, r2),
                file_name="clv_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting CSV upload...")
