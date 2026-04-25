import yfinance as yf
import pandas as pd

nvidia = yf.Ticker("NVDA")
hist = nvidia.history(period="5y")

df = pd.DataFrame()
df['Date'] = hist.index.tz_localize(None)
df['Open'] = hist['Open'].values
df['High'] = hist['High'].values
df['Low'] = hist['Low'].values
df['Close'] = hist['Close'].values
df['Volume'] = hist['Volume'].values

df.to_excel('data_NVDA.xlsx', index=False)
print(f"data_NVDA.xlsx created with {len(df)} rows.")
