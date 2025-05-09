# Main Streamlit application file 

import streamlit as st
from chatbot_core import Chatbot, OPENAI_API_KEY

# Page configuration
st.set_page_config(page_title="NADO AIChatbot", page_icon="🤖", layout="centered")

# Title for the app
st.title("NADO - Your Health Provider Assist Consultant 💬")

# Initialize chatbot instance in session state
if "chatbot_instance" not in st.session_state:
    st.session_state.chatbot_instance = Chatbot()

bot = st.session_state.chatbot_instance

# Initialize chat history in session state
if "messages" not in st.session_state:
    # Extracting the default opening question from SYSTEM_PROMPT
    # This is a simple way; more robust parsing might be needed if SYSTEM_PROMPT structure changes significantly.
    # try:
    #     opening_question = SYSTEM_PROMPT.split('Your default opening question is: "')[1].split('"')[0]
    # except IndexError:
    #     # Fallback if parsing fails
    #     opening_question = "Hi 👋 How can I assist you today with your NDIS provider journey?"
    opening_question = bot.get_opening_question()
    st.session_state.messages = [{"role": "assistant", "content": opening_question}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What can I help you with today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Get response from chatbot
        # Check if OPENAI_API_KEY is set (via bot.client)
        if not bot.client: # client is None if API key was not found during Chatbot initialization
             full_response = "Sorry, the chatbot is not configured with an API key. I cannot process your request at the moment."
        else:
            full_response = bot.get_response(prompt)
        
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Instructions for running the app (optional, can be displayed in the UI)
st.sidebar.info(
    "**About NADO:**\n"
    "NADO is your friendly success consultant for Health Provider Assist. \n"
    "Ask about starting or growing your NDIS business!"
)

if not OPENAI_API_KEY:
    st.error("Warning: OPENAI_API_KEY environment variable not set. The chatbot may not function correctly.")

