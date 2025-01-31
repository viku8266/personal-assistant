import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader, UnstructuredImageLoader
from custom_loaders import VideoTranscriptionLoader

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import shutil

from vector_store_manager import TextFileVectorStore
from llm_client import LLMClient
# Load environment variables
load_dotenv()
 
class DocumentQA:
    def __init__(self):
        self.llm = LLMClient().llm
        self.chat_history = []
        self.vector_store_manager = TextFileVectorStore()
        self.vector_store_manager.load_vector_store('src/python/vector_store-v2.faiss')
        # Initialize QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store_manager.get_vector_store().as_retriever(),
            return_source_documents=False
        )

    def ask_question(self, question: str) -> str:
        context = self.vector_store_manager.fetch_data(query=question)
        system_context = f"You are a helpful assistant that can answer questions about the documents in the directory. use following context to answer the question: ```{context}``` If you don't know the answer, say 'I don't know'."
        result = self.qa_chain.invoke({"question": question, "chat_history": self.chat_history, "system_context": system_context})
        print(result["answer"].split("</think>")[0].split("<think>")[1])
        answer = result["answer"].split("</think>")[1]
        # Update chat history
        self.chat_history.append((question, answer))
        return answer

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
