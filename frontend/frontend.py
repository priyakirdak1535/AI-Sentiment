import streamlit as st
import requests

# Title
st.title("💬 Sentiment Analyzer")
st.write("Enter text and check if it's Positive or Negative")

# Input box
text = st.text_area("Enter your text here:")

# Button
if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()["sentiment"]

                if "Positive" in result:
                    st.success(result)
                else:
                    st.error(result)
            else:
                st.error("API Error ❌")

        except:
            st.error("Could not connect to API 🚫")