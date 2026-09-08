import streamlit as st
import os

from rag_utility import document_ingestion, retrieval

working_dir=os.path.dirname(os.path.abspath(__file__))
st.title("Multi PDF question answering Chatbot")

uploaded_files=st.file_uploader("Upload Pdfs",type=['pdf'],accept_multiple_files=True)

if uploaded_files is not None:
    file_names=[]
    for uploaded_file in uploaded_files:
        save_path=os.path.join(
            working_dir,
            uploaded_file.name
        )
        with open(save_path,'wb') as f:
            f.write(uploaded_file.getbuffer())

        file_names.append(uploaded_file.name)


    if st.button("Process PDFs"):
        document_ingestion(file_names)
        st.success("documents processed successfully")

question=st.text_area("Question Answering")

if st.button("Answer"):
    if question is not None:
        answer=retrieval(question)
        st.markdown(answer)
