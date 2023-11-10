import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI


#fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors


    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = OpenAI(temperature=0.7)

        os.environ["SERPAPI_API_KEY"] = "fe1234c85a82ac23156439b6d5a1fc665922cf40e632882b9bae8a13360dda9c"
       
        tools = load_tools(["serpapi"], llm=llm)

        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)

        response = agent({"input": query})

        google_search = (response["output"])

        len_history = len(st.session_state["history"])

        qa_template = """
        You are a financial expert with crypto market experience.
        Using the chatgpt to answer the question if no relevant context is found.
        Include the sentence "Disclaimer: The crypto market is risky, investing should be approached cautiously" with a new line at the end if and only if the question is related to investment.
        context: {context}
        google_search:{google_search}
        =========
        question: {question}
        ======
        """

        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "google_search",  "question" ])

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"], "google_search": google_search}

        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))

        if len_history < 5:
        
            return result["answer"]

        else:

            return  ("Thank you for your participation！ We will limit the test to 5 times. We hope you will learn more about aivestDAO’s service.")

