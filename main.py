import streamlit as st
from get_data import fetch_data
from data_visualizations import model_price, volatility_analysis

stock_options= ['','MSFT', 'INTU', 'IDXX', 'NEE', 'NVDA']

st.title("AI-Powered Market Analytics")
selected_stock= st.selectbox("Select a stock: ", stock_options)

#fetch data
try:
    if(selected_stock==''):
        st.write("Select a stock")
    df=fetch_data(selected_stock)
    st.write(f"Displaying data for {selected_stock}")

    #show visualizations
    fig=model_price(selected_stock)
    st.plotly_chart(fig)

    fig2=volatility_analysis(selected_stock)
    st.plotly_chart(fig2)
except:
    if(selected_stock!=''):
        st.write(f"Data for {selected_stock} not available")


