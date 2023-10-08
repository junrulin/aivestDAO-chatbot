import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
import streamlit as st


#To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

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

    # uploaded_file = utils.handle_upload(["pdf", "txt", "csv"])

    # Configure the sidebar
    sidebar.show_options()
    sidebar.about()

    # Initialize chat history
    history = ChatHistory()
    try:
        with open('./static/胡子观币.txt','rb') as f:
            uploaded_file = f
            chatbot = utils.setup_chatbot(
                        # uploaded_file, st.session_state["model"], st.session_state["temperature"]
                        f, st.session_state["model"], st.session_state["temperature"]
                    )
        st.session_state["chatbot"] = chatbot
        
        st.markdown(
            "<script>alert('111')var cssId = 'myCss';if (!document.getElementById(cssId)){    var head  = document.getElementsByTagName('head')[0];    var link  = document.createElement('link');    link.id   = cssId;    link.rel  = 'stylesheet';    link.type = 'text/css';    link.href = 'http://website.example/css/stylesheet.css';    link.media = 'all';    head.appendChild(link);}</script>",
            unsafe_allow_html=True,
        )

        if st.session_state["ready"]:

            # Create containers for chat responses and user prompts
            response_container, prompt_container = st.container(), st.container()

            with prompt_container:
                # Display the prompt form
                is_ready, user_input = layout.prompt_form()
                
                # Initialize the chat history
                history.initialize(uploaded_file)

                # Reset the chat history if button clicked
                if st.session_state["reset_chat"]:
                    history.reset(uploaded_file)

                if is_ready:
                    # Update the chat history and display the chat messages
                    history.append("user", user_input)

                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()

                    output = st.session_state["chatbot"].conversational_chat(user_input)

                    sys.stdout = old_stdout

                    history.append("assistant", output)

                    # Clean up the agent's thoughts to remove unwanted characters
                    thoughts = captured_output.getvalue()
                    cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                    cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                    # Display the agent's thoughts
                    with st.expander("Display the agent's thoughts"):
                        st.write(cleaned_thoughts)

            history.generate_messages(response_container)
    except Exception as e:
        st.error(f"Error: {str(e)}")
