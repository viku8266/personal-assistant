from vector_store_manager import TextFileVectorStore

vector_store_manager = TextFileVectorStore()
vector_store_manager.load_vector_store("vector_store-v1.faiss")
vector_store_manager.add_java_files_to_vector_store("/Users/vikasvashistha/github/personal-assistant/code_files/paytm-disbursal-platform")
vector_store_manager.save_vector_store("vector_store-v1.faiss")
