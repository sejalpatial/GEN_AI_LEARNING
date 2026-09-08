import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


#load environment variables from .env file
load_dotenv()

#Find the folder in which this Python file is located, and save that folder's path in working_dir
working_dir=os.path.dirname(__file__)  #__file__ -> gives path of current file , os.path.dirname() gives the directory name of current file

#it is useful beacuse we need not to hardcode the path , if we do so then if in future the path of project chanegs the hardcoded path would give an error as the project is no lnger stored in that location therefore the code breaks

#load the embedding model
embedding=HuggingFaceEmbeddings()


#load the llm
llm=ChatGroq(
    model='allam-2-7b',
    temperature=0.0
)

#document ingestion function
def process_document_to_chroma_db(file_name):
    #load the pdf document using unstructuredpdfloader
    loader=UnstructuredPDFLoader(f"{working_dir}/{file_name}") #we are concatenating the path of file with the working directory file path
    documents=loader.load()

    #split the document into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts=text_splitter.split_documents(documents)
    #store the document in chromadb after embedding
    vectordb=Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore" #tells Chroma where to save the vector database on your computer
    )
    return 0

def answer_question(user_question):
    #load the persistent Chroma vector database
    vectordb=Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    #create a retriever for document search
    retriever=vectordb.as_retriever(
        search_kwargs={"k":5}
    )

    #create a RetrievalQA chain to answer user questions
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    response=qa_chain.invoke({"query": user_question}) #this automatically embeds the questions and performs similarity search
    answer=response["result"]
    source_documents=response["source_documents"]
    return answer,source_documents


