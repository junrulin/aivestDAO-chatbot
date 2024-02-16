import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="aivestDAO | Chat-Bot 🤖", initial_sidebar_state="collapsed")


#Contact
with st.sidebar.expander("📬 Contact"):
    st.write("**Mail** : zzmpp@yahoo.com")
    st.write("**Created by Zheng**")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>aivestDAO-chat, your web3 financial investment assistant 🤖</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>I'm aivestDAO-chat, an intelligent chatbot created by combining 
    the strengths of Large Language Model, Langchain and Streamlit.</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")


#aivestDAO's Pages
st.subheader("🚀 aivestDAO's Pages")
st.write("""
- **aivestDAO-chat**: General Chat on private data with a [vectorstore](https://github.com/facebookresearch/faiss) | works with [ConversationChain](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html)
""")
st.markdown("---")


#Contributing
st.markdown("### 🎯 Contributing")
st.markdown("""
**aivestDAO is under regular development. Feel free to contribute and help me make it even more data-aware!**
""", unsafe_allow_html=True)

st.markdown(
    '<link rel="stylesheet" type="text/css" href="./static/index.css">',
    unsafe_allow_html=True,
)