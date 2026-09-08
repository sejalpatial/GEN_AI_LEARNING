import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

working_dir=os.path.dirname(__file__)

embedding=HuggingFaceEmbeddings()

llm=ChatGroq(
    model='allam-2-7b',
    temperature=0.0
)

def document_ingestion(file_name):
    all_documents=[]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    #whole process runs for number of pdfs attached times
    for f in file_name:  #ex: many files are there like abc.pdf,xyz,pdf
        loader=UnstructuredPDFLoader( os.path.join(working_dir, f))  #whatever pdf you should load is present at this location
        documents=loader.load() #actual loading ,for same pdf diffrent pages are divided into documents , ex: now documents = [Document(page_content="AI is ...", metadata={...}),Document(page_content="Neural networks ...", metadata={...})]

        for doc in documents: #go through every document extracted from this pdf
            doc.metadata["source"]=f #this adds pdf filename to metadata of every document ex: Document 1: content :"ai is..",metadata={"source":"Ai.pdf"} document 2 content="ml is...", metadata={"source":ai.pdf} we do it beacuse when retriever finds a relevant chunk we know that from which pdf it is extracted from
        all_documents.extend(documents) #collecting all documents from pdfs into one list to create one vector database for all pdfs
#--------------------------------------------------------------------------

        texts=text_splitter.split_documents(all_documents)
        vector_db=Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=os.path.join(
            working_dir,
            "doc_vectorstore"
        )
        )
        return 0

def retrieval(question):
        vector_db=Chroma(
            persist_directory=os.path.join(working_dir,"doc_vectorstore"),
            embedding_function=embedding
        )
        retriever=vector_db.as_retriever()

        qa_chain=RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        response=qa_chain.invoke(question)
        answer=response["result"]
        return answer




