import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader, UnstructuredImageLoader
from custom_loaders import VideoTranscriptionLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import shutil

# Load environment variables
load_dotenv()

class DocumentQA:
    def __init__(self, docs_dir: str, model_name: str = "mixtral-8x7b-32768"):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.llm = ChatGroq(temperature=0.1, model=self.model_name, groq_api_key="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")
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = []

    def check_status(self) -> dict:
        """Check the status of core components"""
        status = {
            "llm_connected": True,
            "embeddings_ready": True,
            "documents_loaded": self.vector_store is not None,
            "qa_chain_ready": self.qa_chain is not None
        }
        
        try:
            # Verify LLM connection
            self.llm.invoke("test")
        except Exception:
            status["llm_connected"] = False
            
        try:
            # Verify embeddings
            self.embeddings.embed_query("test")
        except Exception:
            status["embeddings_ready"] = False
            
        return status

    def load_documents(self):
        """Load documents from the specified directory with support for multiple file types"""
        # Configure loaders for different file types
        loaders = [
            DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.pdf", loader_cls=PDFMinerLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.jpg", loader_cls=UnstructuredImageLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.jpeg", loader_cls=UnstructuredImageLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.png", loader_cls=UnstructuredImageLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.mp4", loader_cls=VideoTranscriptionLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.avi", loader_cls=VideoTranscriptionLoader),
            DirectoryLoader(self.docs_dir, glob="**/*.mov", loader_cls=VideoTranscriptionLoader)
        ]
        
        # Load documents from all supported file types
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading some files: {str(e)}")
                continue
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Initialize QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )

    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer based on the document context"""
        if not self.qa_chain:
            raise ValueError("Documents haven't been loaded. Call load_documents() first.")
        
        result = self.qa_chain({"question": question, "chat_history": self.chat_history})
        answer = result["answer"]
        
        # Update chat history
        self.chat_history.append((question, answer))
        return answer

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []

    def handle_file_upload(self, files):
        """Handle file upload and document processing"""
        try:
            # Clear existing documents
            if os.path.exists(self.docs_dir):
                shutil.rmtree(self.docs_dir)
            os.makedirs(self.docs_dir)

            # Save uploaded files
            for file in files:
                file_path = os.path.join(self.docs_dir, os.path.basename(file.name))
                shutil.copy2(file.name, file_path)

            # Load and process documents
            self.load_documents()
            return "Documents uploaded and processed successfully!"
        except Exception as e:
            return f"Error processing documents: {str(e)}"

def create_gradio_interface():
    # Available Groq models
    available_models = [
        "mixtral-8x7b-32768",
        "llama2-70b-4096",
        "gemma-7b-it"
    ]
    
    qa_system = DocumentQA("documents", model_name=available_models[0])
    
    def update_model(model_name):
        qa_system.model_name = model_name
        qa_system.llm = ChatGroq(temperature=0.1, model=model_name, groq_api_key=os.getenv("GROQ_API_KEY"))
        return f"Model updated to {model_name}"

    def check_app_status():
        status = qa_system.check_status()
        status_text = "\n".join([
            f"✓ LLM Connected: {status['llm_connected']}",
            f"✓ Embeddings Ready: {status['embeddings_ready']}",
            f"✓ Documents Loaded: {status['documents_loaded']}",
            f"✓ QA Chain Ready: {status['qa_chain_ready']}"
        ])
        return status_text

    def process_question(question, history):
        try:
            answer = qa_system.ask_question(question)
            history.append((question, answer))
            return "", history
        except Exception as e:
            return "", history + [(question, f"Error: {str(e)}")]

    def upload_files(files):
        return qa_system.handle_file_upload(files)

    def clear_history():
        qa_system.clear_chat_history()
        return None

    with gr.Blocks(title="Document Q&A Assistant") as interface:
        gr.Markdown("# Document Q&A Assistant")
        
        # Add model selection dropdown
        model_dropdown = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="Select Model",
            interactive=True
        )
        
        # Add status indicator
        status_box = gr.Textbox(
            label="Application Status",
            value="Click 'Check Status' to view the application status",
            interactive=False
        )
        status_button = gr.Button("Check Status")
        with gr.Row():
            with gr.Column(scale=2):
                file_output = gr.File(
                    file_count="multiple",
                    label="Upload Documents (Supported: TXT, PDF, JPG, JPEG, PNG, MP4, AVI, MOV)"
                )
                upload_button = gr.Button("Process Documents")

        chatbot = gr.Chatbot(height=400)
        question = gr.Textbox(label="Ask a question about your documents", placeholder="Type your question here...")
        clear_button = gr.Button("Clear Chat History")

        # Set up event handlers
        model_dropdown.change(
            update_model,
            inputs=[model_dropdown],
            outputs=[gr.Textbox(label="Model Status")]
        )
        status_button.click(
            check_app_status,
            outputs=[status_box]
        )
        upload_button.click(
            upload_files,
            inputs=[file_output],
            outputs=[gr.Textbox(label="Upload Status")]
        )
        question.submit(
            process_question,
            inputs=[question, chatbot],
            outputs=[question, chatbot]
        )
        clear_button.click(
            clear_history,
            outputs=[chatbot]
        )

    return interface

def main():
    interface = create_gradio_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()