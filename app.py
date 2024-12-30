from streamlit_cookies_controller import CookieController
import streamlit as st
from llm_utils import process
import os
os.environ["GROQ_API_KEY"] = st.secrets['GROQ_API_KEY']


controller = CookieController()

# user_id = st.text_input("Enter your User ID")

message = st.text_input("Enter your message")

if st.button("Send"):
    process(message, controller, st)
