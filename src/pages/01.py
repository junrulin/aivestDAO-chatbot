import os
import json
import pandas as pd
import streamlit as st
from tinydb import TinyDB
from streamlit_feedback import streamlit_feedback
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from streamlit.components.v1 import html

def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)


#To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


@st.cache_resource
def print_first_assistant_question():
    assistant_message = "Hello ! Please enter your wallet address for the analysis and recommendation 🤗"
    with st.chat_message("assistant"):
        st.markdown(assistant_message)

history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="💬", page_title="aivestDAO | Chat-Bot 🤖", initial_sidebar_state="collapsed")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header("PDF, TXT, CSV")


# user_api_key = utils.load_api_key()
user_api_key = "sk-XgSkVM4ucuNc2n2rDc0e38B3D90544B0Bb32167a565aCe09"
user_api_base = "https://api.zeroai.link/v1"

if not user_api_key:
    layout.show_api_key_missing()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key
    os.environ["OPENAI_API_BASE"] = user_api_base

    # Configure the sidebar
    sidebar.show_options()
    sidebar.about()

    # Initialize chat history
    history = ChatHistory()
    # Initialize chat history
    db = TinyDB('chat_history.json')

    # Initialize chat history from previous sessions
    chat_history = db.all()


    try:
        with open('./static/胡子观币.txt','rb') as f:
            uploaded_file = f
            chatbot = utils.setup_chatbot(
                        f, st.session_state["model"], st.session_state["temperature"]
                    )
        st.session_state["chatbot"] = chatbot

        st.markdown(
            "<script>alert('111')var cssId = 'myCss';if (!document.getElementById(cssId)){    var head  = document.getElementsByTagName('head')[0];    var link  = document.createElement('link');    link.id   = cssId;    link.rel  = 'stylesheet';    link.type = 'text/css';    link.href = 'http://website.example/css/stylesheet.css';    link.media = 'all';    head.appendChild(link);}</script>",
            unsafe_allow_html=True,
        )

        if st.session_state["ready"]:

                print_first_assistant_question()

                for message in chat_history:
                    with st.chat_message(message["role"]):
                        if message['role'] == "assistant":
                            message_content = message["content"].split('#')
                          #  st.write(message_content)
                            if len(message_content)>1:
                                st.markdown(message_content[0])
                                df_plot = pd.DataFrame.from_dict(json.loads(message_content[1]))
                                df_plot.index = pd.to_datetime(df_plot.index, unit='ms')
                                st.line_chart(df_plot)
                                st.markdown(message_content[2])
                                st.markdown(message_content[3])
                            else:
                                st.markdown(message["content"])
                        else:
                            st.markdown(message["content"])

                user_input = st.chat_input("I'm aivestDAO, let me share my insights!")

                if user_input:
                    # Update the chat history and display the chat messages
                    user_message = {"role": "user", "content": user_input}
                    chat_history.append(user_message)
                    db.insert(user_message)
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    output = st.session_state["chatbot"].conversational_chat(user_input)
                    assistant_message = {"role": "assistant", "content": output}
                    chat_history.append(assistant_message)
                    db.insert(assistant_message)
                    feedback = streamlit_feedback(
                            feedback_type="thumbs"
                    )

        if len(db.all()) > 0:
                    # Show button if Chat history not empty
            if st.button("Reset Chat"):
                        # Delete all entries in Database
                db.truncate()
                        # Reload Page
                st.rerun()



    except Exception as e:
        st.error(f"Error: {str(e)}")
