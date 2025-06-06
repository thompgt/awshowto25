import streamlit as st
import pandas as pd
import numpy as np
from get_data import fetch_data
from data_visualizations import model_price, volatility_analysis
from e_genai_oneshot import gen_insight

stock_options = ['','MSFT', 'INTU', 'IDXX', 'NEE', 'NVDA']

st.title("AI-Powered Market Analytics")
selected_stock = st.selectbox("Select a stock: ", stock_options)

try:
    if selected_stock == '':
        st.write("Select a stock")
    else:
        df = fetch_data(selected_stock)
        st.write(f"Displaying data for {selected_stock}")

        fig = model_price(selected_stock)
        st.plotly_chart(fig)

        fig2 = volatility_analysis(selected_stock)
        st.plotly_chart(fig2)
        
        st.subheader("AI-Powered Volatility Classification")
        
        try:
            stock_data = pd.read_csv(f'{selected_stock}.csv')
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)

            if 'short_term_volatility' in stock_data.columns:
                current_volatility = stock_data['short_term_volatility'].dropna().iloc[-1]
                volatility_type = "Short-term (30-day)"
            elif '3_Year_Volatility' in stock_data.columns:
                current_volatility = stock_data['3_Year_Volatility'].dropna().iloc[-1]
                volatility_type = "Long-term (3-year)"
            
            st.write(f"**Current {volatility_type} Volatility: {current_volatility:.3f} ({current_volatility*100:.1f}%)**")
            
            with st.spinner("Getting AI classification..."):
                classification_result = gen_insight(current_volatility)
            
            st.success(classification_result)
                        
        except Exception as volatility_error:
            st.error(f"Could not extract volatility for AI classification: {str(volatility_error)}")
            
except Exception as e:
    if selected_stock != '':
        st.error(f"Data for {selected_stock} not available: {str(e)}")
        
st.sidebar.header("About Volatility Classification")
st.sidebar.write("""
This application uses AWS Bedrock AI to classify stock volatility.

**Volatility Ranges:**
- **Low Volatility**: < 20%
- **Medium Volatility**: 20% - 40%  
- **High Volatility**: > 40%

The AI model analyzes the volatility value and provides a classification with insights.
""")

st.sidebar.header("Available Stocks")
for stock in stock_options[1:]:  # Skip empty string
    st.sidebar.write(f"• {stock}")