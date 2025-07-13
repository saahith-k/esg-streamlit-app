import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("rf_model.pkl")

# Page config
st.set_page_config(
    page_title="ESG Sustainability Predictor",
    page_icon="🌿",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/ESG_logo.svg/640px-ESG_logo.svg.png", width=180)
    st.title("🌿 ESG App")
    st.info("Check if a company is sustainable based on ESG scores.\n\nYou can input scores manually or upload a CSV.")

# Landing Title
st.title("🌱 ESG Sustainability Predictor")
st.markdown("""
Welcome to the **ESG Sustainability Predictor**.  
This tool uses a trained **Random Forest model** to determine if a company is **Sustainable** or **Not Sustainable** based on its ESG (Environmental, Social, Governance) scores.
""")

st.markdown("---")

# 🎯 Manual Input
st.header("🧪 Predict for One Company")
with st.form("input_form"):
    env_score = st.number_input("🌍 Environment Score", min_value=0, max_value=1000, value=500)
    soc_score = st.number_input("👥 Social Score", min_value=0, max_value=1000, value=300)
    gov_score = st.number_input("🏛️ Governance Score", min_value=0, max_value=1000, value=200)
    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_df = pd.DataFrame({
        'environment_score': [env_score],
        'social_score': [soc_score],
        'governance_score': [gov_score]
    })

    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This company is **Sustainable**.")
    else:
        st.error("❌ This company is **Not Sustainable**.")

st.markdown("---")

# 📁 CSV Upload Section
st.header("📁 Batch Prediction (Upload CSV)")
st.markdown("Upload a CSV with `environment_score`, `social_score`, `governance_score` columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        input_cols = ['environment_score', 'social_score', 'governance_score']
        if all(col in df.columns for col in input_cols):
            df['Prediction'] = model.predict(df[input_cols])
            df['Prediction Label'] = df['Prediction'].map({1: '✅ Sustainable', 0: '❌ Not Sustainable'})
            st.success("✔️ Predictions completed!")
            st.dataframe(df)
        else:
            st.error("⚠️ CSV must contain columns: environment_score, social_score, governance_score")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")

# 📊 Feature Importance
st.header("📈 Feature Importance")
importance = pd.Series(model.feature_importances_, index=['Environment', 'Social', 'Governance'])

fig, ax = plt.subplots()
importance.sort_values().plot(kind='barh', color='seagreen', ax=ax)
plt.title("Random Forest Feature Importance")
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Made with 💚 by Saahith • Powered by Streamlit")
