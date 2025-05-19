import gradio as gr
from chat import answer_query

def chat_fn(user_input):
    return answer_query(user_input)

iface = gr.Interface(fn=chat_fn, 
                     inputs="text", 
                     outputs="text",
                     title="Home Device Manual Assistant",
                     description="Ask how to configure or troubleshoot your devices.")
iface.launch()
