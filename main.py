import streamlit as st
from get_data import fetch_data
from data_visualizations import model_price, volatility_analysis

stock_options= ['IDXX', 'INTU', 'MSFT', 'NEE', 'NVDA']

st.title("AI-Powered Market Analytics")
selected_stock= st.selectbox("Select a stock: ", stock_options)

#fetch data
df=fetch_data(selected_stock)
st.write(f"Displaying data for {selected_stock}")

#show visualizations
fig=model_price(selected_stock)
st.plotly_chart(fig)

fig2=volatility_analysis(selected_stock)
st.plotly_chart(fig2)
