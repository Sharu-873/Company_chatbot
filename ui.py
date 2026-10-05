import streamlit as st

from app import answer
# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🤖 Novatech Robo Chatbot")

st.write(
    "Ask anything about the company, courses, training programs, school events, college events, humanoid robot, drones, or Robofest."
)


# -----------------------
# CHAT HISTORY
# -----------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# USER INPUT
# -----------------------

user_input = st.chat_input("Ask your question")

if user_input:

    # Store user message
    st.session_state.history.append(
        ("You", user_input)
    )

    # Generate response
    response = answer(user_input)

    # Store bot response
    st.session_state.history.append(
        ("Bot", response)
    )

# -----------------------
# DISPLAY CHAT
# -----------------------

for role, msg in st.session_state.history:

    if role == "You":

        with st.chat_message("user"):
            st.markdown(msg)

    else:

        with st.chat_message("assistant"):
            st.markdown(msg)