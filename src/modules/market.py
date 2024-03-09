import pandas as pd
import requests
import streamlit as st

class Markets:

    @st.cache_data
    def market_data(ttl = '1d'):

        coin_list = ['bitcoin-cash','the-open-network','polkadot','solana','cardano','chainlink','ripple','matic-network','bitcoin','tron','internet-computer','shiba-inu','dogecoin','litecoin','binancecoin','avalanche-2','ethereum','staked-ether','immutable-x','uniswap']

        df_price_list = []

        df_cap_list = []

        for coin_name in coin_list:

            url = "https://api.coingecko.com/api/v3/coins/"+coin_name+"/market_chart?vs_currency=usd&days=30&interval=daily"

            headers = {"x-cg-demo-api-key": "CG-4wWNWn4ZMRasm9qs211cobTr"}

            response = requests.get(url, headers=headers)

            coin_price_list =  ((response.json()['prices']))

            df_price = pd.DataFrame(coin_price_list, columns = ['Date', coin_name])

            df_price['Date'] = pd.to_datetime(df_price['Date'],unit='ms')

            df_price =  df_price.set_index('Date')

            df_price_list.append(df_price)

            coin_cap_list =  ((response.json()['market_caps']))

            df_cap = pd.DataFrame(coin_cap_list, columns = ['Date', coin_name])

            df_cap['Date'] = pd.to_datetime(df_cap['Date'],unit='ms')

            df_cap =  df_cap.set_index('Date')

            df_cap_list.append(df_cap)

        df_price_final = pd.concat(df_price_list, axis=1, join='inner')

        df_cap_final = pd.concat(df_cap_list, axis=1, join='inner')

        index_name = "alongside-crypto-market-index"

        url = "https://api.coingecko.com/api/v3/coins/"+index_name+"/market_chart?vs_currency=usd&days=30&interval=daily"

        headers = {"x-cg-demo-api-key": "CG-4wWNWn4ZMRasm9qs211cobTr"}

        response = requests.get(url, headers=headers)

        index_price_list =  ((response.json()['prices']))

        df_index_price = pd.DataFrame(index_price_list, columns = ['Date', index_name])

        df_index_price['Date'] = pd.to_datetime(df_index_price['Date'],unit='ms')

        df_index_price =  df_index_price.set_index('Date')

        return (df_price_final, df_cap_final, df_index_price)