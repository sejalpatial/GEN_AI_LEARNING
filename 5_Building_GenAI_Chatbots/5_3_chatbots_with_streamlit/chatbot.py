from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

#load the env variables
load_dotenv()

#streamlit page setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🗪",
    layout="wide",
)
st.title("🗪Gen AI Chatbot")

# chat_history=[]  #if we use this list , then everytime user clicks on a button or intercats with ui the entire page is loaded again therefore chat history also gets refreshed and loses data
#therefore , we use session_state which makes sure that data inside it , is not lost
#Unlike traditional websites,
#  Streamlit reruns your script from top to bottom every time the user interacts with the page.
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]


# as we know , everytime user insteracts with ui , the entire page reruns so to make the prev chat appear in ui we loop and display it 
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):  #with creates a context. Everything inside it belongs to that chat message.
        st.markdown(message['content'])

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0 #A parameter that controls randomness in the model's responses. Lower values make outputs more deterministic, while higher values make them more creative.
)
user_prompt=st.chat_input("Ask chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user", "content":user_prompt})

    response=llm.invoke(
        input=[{"role":"system","content":"you are a helpful assistant"}, *st.session_state.chat_history]
    )
    assistant_response=response.content
    st.session_state.chat_history.append({"role":"assistant","content":response.content})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)