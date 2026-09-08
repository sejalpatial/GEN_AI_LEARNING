import os
import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question

#set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title ("Document Question Answering RAG")

#file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

if uploaded_file is not None:
    #define save path
    save_path=os.path.join(working_dir,uploaded_file.name)
    #save the file
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    process_document=process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed Successfully")

user_question=st.text_area("Question Answering")

if st.button("Answer"):

    if user_question:

        answer, source_documents = answer_question(user_question)

        st.markdown(answer)

        st.subheader("Sources")

        for doc in source_documents:
            source = doc.metadata.get("source", "Unknown")
            st.write(source)

    else:
        st.warning("Please enter a question.")



