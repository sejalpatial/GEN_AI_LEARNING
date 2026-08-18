from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

st.set_page_config(
    page_title=("Multi-Model-Chatbot"),
    layout="wide"
)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1 {
    text-align: center;
    color: #4CAF50;
}

div[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    width: 70%;
}

.stSelectbox label {
    font-weight: bold;
    color: white;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
        🤖 Multi Model GenAI Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("⚙️ Settings")

    selected_model = st.selectbox(
        "Choose LLM",
        (
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma3"
        )
    )

    st.divider()

    st.success(f"Current Model:\n\n**{selected_model}**")

st.info(f"🚀 Currently using **{selected_model}**")

if "chat_history" not in st.session_state:   #st.seesion_state saves info for one session , so that when the entire app reruns the data is not lost 
    st.session_state.chat_history=[]

if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model

if st.session_state.current_model != selected_model:
    st.session_state.chat_history = []
    st.session_state.current_model = selected_model

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if selected_model in ["gemini-2.5-pro","gemini-2.5-flash"]:
    llm=ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.0
    )
elif selected_model in ["llama-3.1-8b-instant","llama-3.3-70b-versatile"]:
    llm=ChatGroq(
        model=selected_model,
        temperature=0.0
    )
else:
    llm=ChatOllama(
        model=selected_model,
        temperature=0.0
    )
user_input=st.chat_input("Ask something")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role":"user","content":user_input})

    with st.spinner("🤖 Thinking..."):
        response = llm.invoke(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                *st.session_state.chat_history,
            ]
        )

    assistant = response.content
    st.session_state.chat_history.append({"role":"assistant","content":assistant})

    with st.chat_message("assistant"):
        st.markdown(assistant)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + LangChain + Gemini + Groq + Ollama")