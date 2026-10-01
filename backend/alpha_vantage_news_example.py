import requests

API_KEY = "YOUR_KEY"

# AlphaVantage NEWS_SENTIMENT example
params = {
    "function": "NEWS_SENTIMENT",
    "tickers": "AAPL",
    "apikey": API_KEY,
}

r = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
print(r.status_code)
print(r.text[:500])

