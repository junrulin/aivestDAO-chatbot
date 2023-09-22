import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

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



        #collect crypto data in real time from tradeview 
        import tvscreener as tvs

        cs = tvs.CryptoScreener()

        df = cs.get()

        df = df[df['Symbol'].str.contains('BINANCE', na = False)]

        df = df[['Symbol','Ask', 'Oscillators Rating', 'Moving Averages Rating', 'MACD Level (12, 26)']].head(20)

        crypto_real_time = df.to_json(orient = "records")




        qa_template = """
        Use the chatgpt to answer the question wuth relavant context, use the google_search to answer real time information if no relavant context is found.
        context: {context}
        crypto_real_time:{crypto_real_time}
        =========
        question: {question}
        ======
        """

        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "crypto_real_time",  "question" ])

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': QA_PROMPT})







        chain_input = {"question": query, "chat_history": st.session_state["history"], "crypto_real_time": crypto_real_time}

        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        
        return result["answer"]


    
    
