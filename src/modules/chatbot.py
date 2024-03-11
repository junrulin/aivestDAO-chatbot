
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import langchain
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import requests
import json
from langchain.callbacks.base import BaseCallbackHandler
from modules.market import Markets

langchain.verbose = False

memory = ConversationBufferMemory(memory_key="history")

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors


    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """

        address = sorted(filter(str.strip, query.split(' ')), key=len, reverse=True)[0]

        if len(address) > 20:

            url = "http://13.212.166.126:3000/v1/poc/getTxHistory?userAddress="+address
            payload = {}
            headers = {
                'User-Agent': 'Apifox/1.0.0',
                'Accept': '*/*',
                'Host': '13.212.166.126:3000',
                'Connection': 'keep-alive'
            }

            try:

                response = requests.request("GET", url, headers=headers, data=payload)

                data = json.loads(response.text)

                data_history = data['historyData']


            except:

                return ('There is something wrong with network, please come back to the conversation later')


            week_fre = len(data_history)

            half = week_fre//2

            data_1 = data_history[:half]

            data_2 = data_history[half:]

            data_1_token = []
            data_2_token = []
            data_1_fre =  []
            data_2_fre =  []

            for x in data_1:
                data_1_token.append(x.split("\t")[1])

                data_1_fre.append(float(x.split("\t")[3]))


            for x in data_2:
                data_2_token.append(x.split("\t")[1])

                data_2_fre.append(float(x.split("\t")[3]))

            token_1_div = (len(set(data_1_token)))

            token_2_div = (len(set(data_2_token)))

            token_div = len(set(data_1_token + data_2_token))

            token_1_fre = (sum(data_1_fre))

            token_2_fre = (sum(data_2_fre))

            '''
            st.write(token_1_fre)

            st.write(token_2_fre)

            st.write(token_1_div)

            st.write(token_2_div)

            st.write(week_fre)

            st.write(token_div)
            '''

            if (week_fre) >= 7:

                risk = 'risky'

                fre = 'frequent'

            elif ((week_fre >= 2) and (week_fre < 7)) :

                risk = 'moderate'

                fre = 'moderate'

            else:

                risk = 'conservative'

                fre = 'nonfrequent'

            if token_div > 5:

                div = 'diversified'

            else:

                div = 'nondiversified'

            if token_2_fre > token_1_fre:

                fre_trend = 'increasing'

            else:

                fre_trend = 'decreasing'

            if token_2_div > token_1_div:

                div_trend = 'increasing'

            else:

                div_trend = 'decreasing'

            if fre == 'frequent':

                fre_str = 'Based on data from past week, you are highly sensitive to market information and price fluctuations, demonstrating strong execution capabilities. You have confidence in the market, are self-assured about your trading strategy, possess courage, are willing to speculate, and can understand data trends. It is crucial for you to maintain mental stability, and you are likely to excel in your endeavors.'

            elif fre == 'moderate':

                fre_str = 'Based on data from past week, you values reducing risks and enhancing returns equally. You are willing to accept modest risks to seek higher long-term returns. A moderate investor may endure a short-term loss of principal and lower degree of liquidity in exchange for long-term appreciation.'

            else:

                fre_str = 'Based on data from past week, you are very prudent, with strict risk management and a clear understanding of your investment approach. Therefore, factors such as a large volume of market information, changing market sentiment, and frequent price fluctuations do not prompt immediate investment decisions. Even if you sometimes envy others profits, you remain unaffected, carefully weighing the correlation between returns and high-risk situations.'

            if fre_trend == 'increasing':

                fre_trend_str = 'As time passes and your understanding of the cryptocurrency market grows, your enthusiasm for investing increases, accompanied by an expanding risk tolerance.'

            else:

                fre_trend_str = 'With time and increasing knowledge of the cryptocurrency market, you become more composed in your approach to investing, exercising greater caution in terms of risk.'

            if div == 'diversified':

                div_str = 'You favor risk diversification, enhancing the likelihood of returns by diversifying coin types to reduce correlation and allocating appropriate quantities.'

            else:

                div_str = 'Your focus is highly concentrated, spending most of your time monitoring the prices, fundamentals, and market information of a few cryptocurrencies. You base your investment decisions on past experiences and your understanding of their price movements.'

            if div_trend == 'increasing':

               div_trend_str = 'Over time, you become more open to investing in new projects and assets, welcoming new knowledge and market information.'

            else:

                div_trend_str = 'As time progresses, you gain clarity on which cryptocurrencies you excel at managing trades for, no longer spending time diversifying into assets you may not fully endorse.'

            investing_style_str = fre_str + ' ' + fre_trend_str + ' ' + div_str + ' ' + div_trend_str


            df_price, df_market_cap, df_index_price = Markets.market_data()
            mu = expected_returns.mean_historical_return(df_price, frequency=30)
            S = risk_models.sample_cov(df_price)
            max_weight = 1.0/min(10, token_div)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            if risk == 'risky':
                raw_weights = ef.max_quadratic_utility()
            elif risk == 'moderate':
                raw_weights = ef.max_sharpe()
            else:
                raw_weights = ef.min_volatility()
            performance = list(ef.portfolio_performance())
            performance = [x for x in performance]
            performance_str = 'The best portfolio recommended by aivest with modern portfolio theory has monthly return of ' + str('{:.0%}'.format(performance[0])) + ' sharpe ratio of ' + str('{:.0%}'.format(performance[2]))
            weights = (pd.Series(raw_weights))
            latest_prices = get_latest_prices(df_price)
            latest_market_cap = get_latest_prices(df_market_cap)
            price_min = df_price.min(axis=0)
            price_max = df_price.max(axis=0)
            max_drawdown = 1 - price_min / price_max
            df = pd.concat([weights, latest_prices, latest_market_cap, mu, max_drawdown], axis=1)
            df.columns = ['Weight', 'Latest_Price', 'Latest_Market_Cap', 'Monthly_Return', 'Max_Drawdown']
            df.index.name = 'Symbol'
            pd.set_option("display.precision", 3)
            df = df.loc[(df['Weight'] >= 0.01)]
            portfolio_max_drawdown = (df['Weight']*df['Max_Drawdown']).sum()

            df_return = expected_returns.returns_from_prices(df_price)
            def weighted_sum(df, weights):
                return df[list(weights)].mul(weights).sum(axis=1) / sum(weights.values())
            portfolio_return = weighted_sum(df_return, dict(zip(df.index,df.Weight)))
            portfolio_price = expected_returns.prices_from_returns(portfolio_return)
            portfolio_price.columns = ['Date','portfolio_price']
            df_index_price = df_index_price/df_index_price['alongside-crypto-market-index'].iloc[0]
            df_total_price = pd.concat([portfolio_price, df_index_price], axis=1)
            df_total_price.columns = ['portfolio_price', 'index_price']
            df_total_price_str = df_total_price.to_json()
            info_str = (df.to_json())
            final_str = 'Thanks for your info! Here is the investing style analyzed by aivest with wallet address above: '+ address +  investing_style_str + '\n\n' + performance_str + ' and max drawdown of '+str('{:.0%}'.format(portfolio_max_drawdown))+ 'Here you can see more details:'+ '\n\n' + info_str + df_total_price_str
            memory.save_context({"input": "hi"}, {"history": final_str})

        if 'launch' in query.lower():

            with st.chat_message("assistant"):
                st.markdown('You have successfully launched your tokenized model! Tell your friends to come to the "Fair Launch" 🛫buy your token, and leverage your crypto investment insights!')
            return (
                'You have successfully launched your tokenized model! Tell your friends to come to the "Fair Launch" 🛫buy your token, and leverage your crypto investment insights!')

        try:

            qa_template = """
            You are a financial expert with crypto market experience.
            If input contains wallet address, output investing style with bullet points with seperate lines in the begining.
            Use all info in the history to answer the input and include monthly return, sharpe ratio and max drawdown in the answer .
            Output the records from the JSON data and format it as a markdown table with Symbol, Weight, Monthly Return, Max Drawdown and convert float to percentage, Latest Price and Latest Market Cap in float.
            Only use the chatgpt to answer the input if no relevant history is found.
            ------------
            history: {history}
            ------------
            input: {input}
            """

            class StreamHandler(BaseCallbackHandler):
                def __init__(self, container, initial_text=""):
                    self.container = container
                    self.text = initial_text

                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    self.text += token
                    self.container.markdown(self.text)

            QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["history", "input"])

            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature, streaming=True)

            chain = ConversationChain(llm=llm, memory=memory, prompt=QA_PROMPT, verbose=False)

            with st.chat_message("assistant"):

                result = chain.run(query, callbacks=[StreamHandler(st.empty())])

                st.session_state["history"].append((query, result))

                if len(address) > 20:

                    st.line_chart(df_total_price)

                    st.markdown(
                            "Next action: Kindly tell me 'Launch', once you are ready to tokenize and fair launch your portfolio model.🚀")

                    st.markdown(
                            "Disclaimer: The crypto market is risky, investing should be approached cautiously.")


                    return (result +'#'+ df_total_price.to_json()+'#'+ "Next action: Kindly tell me 'Launch', once you are ready to tokenize and fair launch your portfolio model.🚀" + "#" + "Disclaimer: The crypto market is risky, investing should be approached cautiously.")

                return result


        except Exception as e:
            st.error(f"Error: {str(e)}")