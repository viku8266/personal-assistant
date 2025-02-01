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

# List of LLM models supported on groq
GROQ_SUPPORTED_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768"
]

class DocumentQA:
    _current_model_index = 0  # Class variable to track current model index

    def __init__(self):
        self.vector_store_manager = TextFileVectorStore()
        self.vector_store_manager.load_vector_store('src/python/vector_store-v4.faiss')
        self.llm = LLMClient(model_name=GROQ_SUPPORTED_MODELS[0]).llm
        self.qa_chain = ConversationalRetrievalChain.from_llm(self.llm, self.vector_store_manager.get_vector_store().as_retriever())
    @classmethod
    def _get_next_model(cls):
        model = GROQ_SUPPORTED_MODELS[cls._current_model_index]
        cls._current_model_index = (cls._current_model_index + 1) % len(GROQ_SUPPORTED_MODELS)
        return model

    def ask_question(self, question: str , history: str) -> str:
        documents  = self.vector_store_manager.fetch_data(query=question)
        context = "\n".join([document.page_content for document in documents])
        system_context = f"You are a helpful assistant that can answer questions about the documents in the directory. use following context to answer the question: ```{context}``` If you don't know the answer, say 'I don't know'."
        result = self.qa_chain.invoke({"question": question, "chat_history": "", "system_context": system_context})
        answer = result["answer"]
        self.change_model() 
        if len(answer.split('</think>')) > 1:
            answer = answer.split('</think>')[1]
        return answer

    def change_model(self):    
        model_name = self._get_next_model()
        print(f"Using model: {model_name}")
        self.llm = LLMClient(model_name=model_name).llm
        self.qa_chain = ConversationalRetrievalChain.from_llm(self.llm, self.vector_store_manager.get_vector_store().as_retriever())

