from qa_assistant import DocumentQA
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from vector_store_manager import TextFileVectorStore
import httpx
llm = ChatGroq(temperature=0.1, groq_api_key="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",http_client=httpx.Client(verify=False))

vector_store_manager = TextFileVectorStore()
vector_store_manager.load_vector_store('vector_store.faiss')
vector_store = vector_store_manager.get_vector_store()
qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )

result = qa_chain.invoke({"question": "What is payouts?", "chat_history": []})
print(result)