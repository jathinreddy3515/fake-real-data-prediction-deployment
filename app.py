import streamlit as st
import joblib

# Load saved model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news article text to check if it is Real or Fake.")

user_input = st.text_area("Enter News Text Below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_vector = vectorizer.transform([user_input])

        # Predict class
        prediction = model.predict(input_vector)[0]

        # Predict probability
        probability = model.predict_proba(input_vector).max()

        st.subheader("Prediction Result:")
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {round(probability * 100, 2)}%")
