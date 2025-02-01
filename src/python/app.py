import gradio as gr
from qa_assistant import DocumentQA

qa_system = DocumentQA()

def process_message(message, history):
    print(f"Question : {message}")
    print(f"History : {history}")
    answer = qa_system.ask_question(question=message,history=history)
    print(f"Answer : {answer}")
    return answer

def create_gradio_interface():
    with gr.Blocks(title ="Bpay Digital Assistant") as interface:
        gr.Markdown("# Bpay Digital Assistant")
        gr.ChatInterface(fn=process_message, type="messages")
    return interface    


def main():
    interface = create_gradio_interface()
    interface.launch(share=True)
if __name__ == "__main__":
    main()