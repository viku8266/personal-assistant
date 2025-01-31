import gradio as gr
from qa_assistant import DocumentQA
    
def create_gradio_interface():
    qa_system = DocumentQA()
    
    def process_question(question, history):
        try:
            answer = qa_system.ask_question(question)
            history.append((question, answer))
            return "", history
        except Exception as e:
            print(f"Error: {str(e)}")
            return "", history + [(question, f"Error: {str(e)}")]

    def clear_history():
        qa_system.clear_chat_history()
        return None

    with gr.Blocks(title="Simple Q&A Interface", css="""
        body { background-color: #f0f4f8; font-family: 'Arial', sans-serif; }
        .gradio-container { max-width: 800px; margin: auto; padding: 20px; }
        .gradio-container h1 { color: #333; }
        .gradio-container .gr-button { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; transition: background-color 0.3s ease; }
        .gradio-container .gr-button:hover { background-color: #0056b3; }
        .gradio-container .gr-chatbot { border: 1px solid #ccc; border-radius: 5px; background-color: white; padding: 10px; }
    """) as interface:
        gr.Markdown("# Simple Q&A Interface")
        
        with gr.Row():
            clear_button = gr.Button("Clear Chat History", size="sm",scale = 0)
            chatbot = gr.Chatbot(min_width=400, height=400)
        with gr.Row():
            question = gr.Textbox(label="Ask a question", placeholder="Type your question here...", lines=1,scale =100)
            submit_button = gr.Button("Ask",scale=1)
            
        # Set up event handler
        submit_button.click(
            process_question,
            inputs=[question, chatbot],
            outputs=[question, chatbot],
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