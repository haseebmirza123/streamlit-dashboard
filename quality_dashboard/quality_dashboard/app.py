import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Set Streamlit config
st.set_page_config(page_title="Product Quality Dashboard", layout="wide")

# Title and intro
st.title("🏭 Product Quality Prediction Dashboard")
st.markdown("This interactive dashboard allows you to input process parameters and predict product quality using the trained CatBoost model.")

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

model = load_model()
df = load_data()

# Get feature names from dataset (excluding target)
feature_names = df.drop("quality", axis=1).columns.tolist()

# Sidebar: user input
st.sidebar.header("🔧 Input Parameters")

user_input = {}
for feature in feature_names:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Convert user input into DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("🔍 Predict Quality Class"):
    prediction = model.predict(input_df)[0]

    # Map prediction to class label and emoji
    class_labels = {
        1: "⚠️ Inefficient",
        2: "🟡 Acceptable",
        3: "✅ Target",
        4: "🗑️ Waste"
    }

    predicted_label = class_labels.get(int(prediction), "❓ Unknown Class")
    st.success(f"📦 Predicted Product Quality Class: **{predicted_label}**")


# Optional: Show dataset preview
with st.expander("📄 View Sample of Cleaned Dataset"):
    st.dataframe(df.head(10))


# Section: Visualizations
st.subheader("📊 Model Visualizations")

# Confusion Matrix
st.markdown("**Confusion Matrix – Tuned CatBoost**")
cm_img = Image.open("catboost (after tuning).png")
st.image(cm_img, caption="Confusion Matrix – Tuned CatBoost")

# ROC Curve
st.markdown("**ROC Curve – CatBoost**")
roc_img = Image.open("catboost ROC.png")
st.image(roc_img, caption="ROC Curve – CatBoost (Macro AUC = 0.99)")

# Feature Importance
st.markdown("**Top 10 Feature Importances**")
fi_img = Image.open("catboost feature importance.png")
st.image(fi_img, caption="Top 10 Important Features – CatBoost")
