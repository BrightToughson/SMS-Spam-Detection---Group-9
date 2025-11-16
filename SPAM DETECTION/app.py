import streamlit as st
import joblib
import re

# --- I. Model and Vectorizer Loader ---
# Use st.cache_resource to load the model and vectorizer only once


@st.cache_resource
def load_model_and_vectorizer():
    """Loads the pre-trained model and TF-IDF vectorizer."""
    import os

    # Search for the joblib files in common locations
    possible_paths = [
        ('main/sms_spam_model.joblib', 'main/tfidf_vectorizer.joblib'),
        ('sms_spam_model.joblib', 'tfidf_vectorizer.joblib'),
        ('./main/sms_spam_model.joblib', './main/tfidf_vectorizer.joblib'),
    ]

    model_path = None
    vectorizer_path = None

    for m_path, v_path in possible_paths:
        if os.path.exists(m_path) and os.path.exists(v_path):
            model_path = m_path
            vectorizer_path = v_path
            break

    if not model_path or not vectorizer_path:
        st.error(
            f"Model or Vectorizer files not found. Searched: {[p[0] for p in possible_paths]}")
        return None, None

    try:
        # Load the model and vectorizer saved after training on the full dataset
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# --- II. Text Preprocessing Function (MUST match the one used during training) ---


def preprocess_text(text):
    """Cleans and preprocesses the input text to match the training data format."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove all characters that are not lowercase letters or spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- III. Prediction Function ---


def predict_spam(message, model, vectorizer):
    """Preprocesses a message, vectorizes it, and returns the prediction label."""
    # 1. Preprocess the message using the same cleaning function
    cleaned_message = preprocess_text(message)

    # 2. Vectorize the message using the loaded, fitted vectorizer
    vectorized_message = vectorizer.transform([cleaned_message])

    # 3. Predict (0 = Ham, 1 = Spam)
    prediction = model.predict(vectorized_message)

    # 4. Return result label
    return "SPAM" if prediction[0] == 1 else "HAM (Not Spam)"

# --- IV. Streamlit App Interface ---


def main():
    st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
    st.title("📩 SMS Spam Detection")
    st.markdown(
        "A model trained to distinguish between legitimate messages (Ham) and unwanted messages (Spam).")
    st.markdown("---")

    # Load the resources
    model, vectorizer = load_model_and_vectorizer()

    if model and vectorizer:

        st.header("Enter an SMS Message")

        # Initialize inputs
        user_input = ""
        uploaded_file = None

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["Type Message", "Upload File"])

        with tab1:
            user_input = st.text_area(
                "Message Input",
                "",
                height=150,
                key="text_input"
            )

        with tab2:
            uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"], key="uploader")
            input_method2 = ""
            if uploaded_file is not None:
                try:
                    input_method2 = uploaded_file.read().decode("utf-8")
                except Exception:
                    input_method2 = str(uploaded_file.read())
                st.text(f"File content preview: {input_method2[:100]}...")

        # Single action button: use uploaded file if present, otherwise typed message
        if st.button("Check if Spam", type="primary"):
            # Prefer uploaded file when available
            if uploaded_file is not None and input_method2:
                message_to_check = input_method2
            elif user_input:
                message_to_check = user_input
            else:
                st.warning("Please enter a message or upload a file to run the prediction.")
                return

            result = predict_spam(message_to_check, model, vectorizer)
            st.markdown("## Prediction Result:")
            if result == "SPAM":
                st.error(f"**{result}**")
                st.markdown(
                    "**⚠️ Warning:** This message is likely a spam or phishing attempt. Stay cautious!")
            else:
                st.success(f"**{result}**")
                st.markdown(
                    "**✓ Safe:** This message appears to be legitimate. You're good to go!")


if __name__ == '__main__':
    main()
