import json
import pandas as pd
import plotly.graph_objects as go
from PatternDetector import CandlestickDetector, ChartPatternDetector

def generate_plotly_pattern_chart(df: pd.DataFrame, ticker: str):
    """
    Generates a Plotly JSON containing a Candlestick chart and pattern annotations.
    """
    if df is None or df.empty:
        return json.dumps({"error": "No data available"})

    # Ensure index is datetime if it's not
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC")])

    annotations = []
    
    # 1. Chart patterns (these use swing highs/lows from the whole df)
    chart_detector = ChartPatternDetector(df)
    chart_patterns = chart_detector.detect_all()
    for p in chart_patterns:
        color = "#00ff00" if p.signal == "BULLISH" else "#ff0000" if p.signal == "BEARISH" else "blue"
        annotations.append(dict(
            x=df.index[-1], y=p.price_at_detection,
            xref="x", yref="y",
            text=f"{p.pattern_type} ({p.signal})",
            showarrow=True, arrowhead=1, ax=-40, ay=-40 if color=="#00ff00" else 40,
            font=dict(color=color, size=11),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=color,
            borderwidth=1
        ))

    # 2. Candlestick patterns (run on a rolling window of last 60 days to show some history)
    lookback = min(60, len(df))
    # We want to avoid overlapping text, so we keep track of y positions per day
    y_offsets = {}
    for i in range(len(df) - lookback, len(df)):
        sub_df = df.iloc[:i+1]
        if len(sub_df) < 20: continue
        
        cd = CandlestickDetector(sub_df)
        patterns = cd.detect_all()
        
        day_date = df.index[i]
        
        for p in patterns:
            if p.confidence > 70:
                color = "#00ff00" if p.signal == "BULLISH" else "#ff0000"
                y_offset = y_offsets.get(day_date, 0)
                ay_base = -30 if color=="#00ff00" else 30
                ay = ay_base + (y_offset * (-15 if color=="#00ff00" else 15))
                y_offsets[day_date] = y_offset + 1
                
                annotations.append(dict(
                    x=day_date, y=p.price_at_detection,
                    xref="x", yref="y",
                    text=p.pattern_type,
                    showarrow=True, arrowhead=2, ax=0, ay=ay,
                    font=dict(color=color, size=9),
                    bgcolor="rgba(0,0,0,0.5)"
                ))

    # Add moving averages to make the chart richer
    if len(df) >= 50:
        ma50 = df['close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color='orange', width=1), name='50 MA'))
    if len(df) >= 200:
        ma200 = df['close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma200, line=dict(color='purple', width=1), name='200 MA'))

    # Add annotations to figure
    fig.update_layout(
        title=f"{ticker} - Technical Patterns (Long Term)",
        yaxis_title="Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        annotations=annotations,
        template="plotly_dark",
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig.to_json()
