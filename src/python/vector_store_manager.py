import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class TextFileVectorStore:
    def __init__(self,embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vector_store = None

    def create_vector_store(self, directory_path: str):
        texts = []
        # Iterate over all text files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        text = f.read()
                        # Split the text into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=15000,
                            chunk_overlap=300,
                            length_function=len
                        )
                        texts.extend(text_splitter.split_text(text))

        # Create vector store
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        print(f"Vector store created for directory {directory_path}")

    def get_vector_store(self):
        if not self.vector_store:
            raise ValueError("Vector store has not been created. Call create_vector_store() first.")
        return self.vector_store

    def upload_file(self, new_directory_path: str):
        # Update the directory path and recreate the vector store
        texts = []
        # Iterate over all text files in the directory
        for root, _, files in os.walk(new_directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        text = f.read()
                        # Split the text into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=15000,
                            chunk_overlap=300,
                            length_function=len
                        )
                        texts.extend(text_splitter.split_text(text))
        self.vector_store.add_texts(texts)

    def fetch_data(self, query: str):
        if not self.vector_store:
            raise ValueError("Vector store has not been created. Call create_vector_store() first.")
        # Fetch data from the vector store based on the query
        results = self.vector_store.similarity_search(query)
        return results

    def save_vector_store(self, file_path: str):
        if not self.vector_store:
            raise ValueError("Vector store has not been created. Call create_vector_store() first.")
        self.vector_store.save_local(file_path)
        print(f"Vector store saved to {file_path}")

    def load_vector_store(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vector store file {file_path} does not exist.")
        self.vector_store = FAISS.load_local(file_path, self.embeddings,allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {file_path}")

    def add_java_files_to_vector_store(self, directory_path: str):
        if not self.vector_store:
            raise ValueError("Vector store has not been created. Call create_vector_store() first.")
        texts = []
        # Iterate over all Java files in the directory
        for root, dirs, files in os.walk(directory_path):
            print(f"indexing root {root} dirs {dirs} files {files} ")
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        text = f.read()
                        # Split the text into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=15000,
                            chunk_overlap=300,
                            length_function=len
                        )
                        texts.extend(text_splitter.split_text(text))
        self.vector_store.add_texts(texts)
        print(f"Java files added to vector store from directory index text length  {len(texts)}")
        




# sample usage
vector_store_manager = TextFileVectorStore()
vector_store_manager.create_vector_store("/Users/vikasvashistha/github/personal-assistant/transcripts")
vector_store_manager.add_java_files_to_vector_store("/Users/vikasvashistha/github/personal-assistant/code_files/paytm-disbursal-platform")
vector_store_manager.save_vector_store("vector_store-v1.faiss")