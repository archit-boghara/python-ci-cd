import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import os

# Set the style for our visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

# Create a directory to save plots if it doesn't exist
output_dir = "financial_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_stock_data(ticker_symbols, period="1y"):
    """
    Fetch stock data for the given ticker symbols over the specified period.
    
    Args:
        ticker_symbols (list): List of stock ticker symbols
        period (str): Time period to fetch data for (e.g., '1d', '1mo', '1y')
    
    Returns:
        DataFrame: Stock closing prices for the given tickers
    """
    data = {}
    
    for ticker in ticker_symbols:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return pd.DataFrame(data)

def calculate_returns(prices_df):
    """
    Calculate daily and cumulative returns from price data.
    
    Args:
        prices_df (DataFrame): DataFrame containing stock prices
    
    Returns:
        tuple: (daily_returns, cumulative_returns)
    """
    # Calculate daily returns
    daily_returns = prices_df.pct_change().dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    return daily_returns, cumulative_returns

def plot_stock_prices(prices_df, title="Stock Prices Over Time"):
    """
    Plot stock prices over time and save to file.
    
    Args:
        prices_df (DataFrame): DataFrame containing stock prices
        title (str): Plot title
    """
    plt.figure(figsize=(12, 7))
    
    for column in prices_df.columns:
        plt.plot(prices_df.index, prices_df[column], label=column, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot instead of showing it
    filename = os.path.join(output_dir, "stock_prices.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Stock prices plot saved to {filename}")

def plot_returns_comparison(cumulative_returns, title="Cumulative Returns Comparison"):
    """
    Plot cumulative returns comparison and save to file.
    
    Args:
        cumulative_returns (DataFrame): DataFrame containing cumulative returns
        title (str): Plot title
    """
    plt.figure(figsize=(12, 7))
    
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column] * 100, label=column, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return (%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot instead of showing it
    filename = os.path.join(output_dir, "cumulative_returns.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Cumulative returns plot saved to {filename}")

def plot_correlation_heatmap(daily_returns, title="Correlation Between Stocks"):
    """
    Plot correlation heatmap between different stocks and save to file.
    
    Args:
        daily_returns (DataFrame): DataFrame containing daily returns
        title (str): Plot title
    """
    corr_matrix = daily_returns.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Save the plot instead of showing it
    filename = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to {filename}")

def plot_volatility(daily_returns, window=20, title="Stock Volatility (Rolling 20-day)"):
    """
    Plot stock volatility using rolling standard deviation and save to file.
    
    Args:
        daily_returns (DataFrame): DataFrame containing daily returns
        window (int): Rolling window size
        title (str): Plot title
    """
    volatility = daily_returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized volatility in percentage
    
    plt.figure(figsize=(12, 7))
    
    for column in volatility.columns:
        plt.plot(volatility.index, volatility[column], label=column, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility (%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot instead of showing it
    filename = os.path.join(output_dir, "volatility.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Volatility plot saved to {filename}")

def main():
    # Define stock tickers to analyze
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    print(f"Fetching data for: {', '.join(tickers)}")
    
    # Get stock price data for the past year
    stock_prices = get_stock_data(tickers, period="1y")
    
    # Calculate returns
    daily_returns, cumulative_returns = calculate_returns(stock_prices)
    
    # Generate visualizations
    print("\nGenerating visualizations...\n")
    
    # Plot 1: Stock prices over time
    plot_stock_prices(stock_prices)
    
    # Plot 2: Cumulative returns comparison
    plot_returns_comparison(cumulative_returns)
    
    # Plot 3: Correlation heatmap
    plot_correlation_heatmap(daily_returns)
    
    # Plot 4: Volatility comparison
    plot_volatility(daily_returns)
    
    print("\nAnalysis complete! All plots saved to the 'financial_plots' directory.")

if __name__ == "__main__":
    main()