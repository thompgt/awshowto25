import pandas as pd
import plotly.graph_objects as go

def model_price(stock_index):
    df=pd.read_csv(f'{stock_index}.csv')
    df['Date']=pd.to_datetime(df.iloc[:, 0])
    df.set_index('Date', inplace=True)

    fig= go.Figure()
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['open'],
                            mode='lines',
                            name='Opening Price',
                            line=dict(color='blue')
                            ))

    fig.update_layout(
        title='Opening Price Over Time',
        xaxis_title="Date",
        yaxis_title="Opening Price $",
        hovermode="x"
    )

    return fig

def volatility_analysis(stock_index):
    df=pd.read_csv(f'{stock_index}.csv')
    df['Date']=pd.to_datetime(df.iloc[:, 0])
    df.set_index('Date', inplace=True)

    fig= go.Figure()

    #long term volatility
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['3_Year_Volatility'],
                            mode='lines',
                            name='Long Term Volatility',
                            line=dict(color='blue')
                            ))
    
    #short term volatility
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['short_term_volatility'],
                            mode='lines',
                            name='Short Term volatility',
                            line=dict(color='red')
                            ))

    fig.update_layout(
        title='Short vs Long Term Volatility over Time',
        xaxis_title="Date",
        yaxis_title="Volatility (Standard Deviation)",
        hovermode="x"
    )

    return fig

