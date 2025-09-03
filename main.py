# quant_analyzer.py
# quant_analyzer.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fetch_data(tickers, period="6mo"):

    print(f"🔄 Fetching data for {tickers}...")
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    closing_prices = data['Close']
    print("✅ Data fetched successfully!")
    return closing_prices


def calculate_returns(price_data):

    print("📊 Calculating daily returns...")
    returns = price_data.pct_change().dropna()
    return returns


def calculate_risk_metrics(returns_data):

    print("⚖️ Calculating risk metrics...")
    avg_annual_return = returns_data.mean() * 252
    annual_volatility = returns_data.std() * np.sqrt(252)
    sharpe_ratio = avg_annual_return / annual_volatility
    return avg_annual_return, annual_volatility, sharpe_ratio


def plot_performance(price_data):

    print("📈 Generating performance chart...")
    plt.figure(figsize=(12, 6))

    normalized_data = (price_data / price_data.iloc[0] * 100)

    for column in normalized_data.columns:
        plt.plot(normalized_data.index, normalized_data[column], label=column, linewidth=2)

    plt.title('Portfolio Performance Comparison (Normalized to 100)')
    plt.xlabel('Date')
    plt.ylabel('Performance (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(returns_data):

    print("🧮 Calculating correlation matrix...")
    plt.figure(figsize=(10, 8))

    corr_matrix = returns_data.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Asset Return Correlations')
    plt.tight_layout()
    plt.show()


def monte_carlo_simulation(ticker, price_data, days=252, simulations=1000):
    print(f"🔮 Running {simulations} Monte Carlo simulations for {ticker}...")

    returns = price_data[ticker].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = price_data[ticker].iloc[-1]

    dt = 1 / days
    results = np.zeros((days, simulations))
    results[0] = last_price

    for t in range(1, days):
        shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=simulations)
        results[t] = results[t - 1] * np.exp(shock)

    return results


def plot_monte_carlo(results, ticker):

    plt.figure(figsize=(12, 6))
    plt.plot(results, linewidth=0.5, alpha=0.1, color='blue')

    # Calculate and plot the mean path
    mean_path = np.mean(results, axis=1)
    plt.plot(mean_path, color='red', linewidth=2.5, label='Mean Path')

    # Calculate and plot confidence intervals
    upper_95 = np.percentile(results, 95, axis=1)
    lower_5 = np.percentile(results, 5, axis=1)
    plt.plot(upper_95, color='darkred', linestyle='--', linewidth=1.5, label='95th Percentile')
    plt.plot(lower_5, color='darkred', linestyle='--', linewidth=1.5, label='5th Percentile')

    plt.title(f'{ticker} Monte Carlo Simulation ({results.shape[1]} paths)')
    plt.xlabel('Trading Days into Future')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Calculate and print some statistics
    final_prices = results[-1]
    expected_price = np.mean(final_prices)
    price_change = ((expected_price - results[0][0]) / results[0][0]) * 100
    confidence_interval = np.percentile(final_prices, [5, 95])

    print(f"\n📈 {ticker} Monte Carlo Results:")
    print(f"Expected price after {len(results)} days: ${expected_price:.2f}")
    print(f"Expected change: {price_change:+.2f}%")
    print(f"95% Confidence Interval: ${confidence_interval[0]:.2f} - ${confidence_interval[1]:.2f}")


if __name__ == "__main__":

    my_portfolio = ['AAPL', 'MSFT', 'GOOG', 'SPY']

    # Fetch and analyze data
    price_data = fetch_data(my_portfolio)
    returns_data = calculate_returns(price_data)
    annual_returns, annual_volatility, sharpe_ratios = calculate_risk_metrics(returns_data)

    # Print report
    print("\n" + "=" * 60)
    print("QUANTITATIVE PORTFOLIO ANALYSIS")
    print("=" * 60)
    for ticker in my_portfolio:
        print(f"{ticker}:")
        print(f"  Avg Annual Return: {annual_returns[ticker]:.2%}")
        print(f"  Annual Volatility: {annual_volatility[ticker]:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratios[ticker]:.2f}")
        print()

    # Visualizations
    plot_performance(price_data)
    plot_correlation_heatmap(returns_data)

    # Monte Carlo simulation for one asset
    mc_results = monte_carlo_simulation('SPY', price_data)
    plot_monte_carlo(mc_results, 'SPY')