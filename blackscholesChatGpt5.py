import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def black_scholes_monte_carlo(S0, X, T, r, sigma, option_type='call', num_simulations=100000):
    dt = T / 252  # Assuming 252 trading days in a year
    num_days = int(T / dt) + 1

    # Analytical Black-Scholes estimate
    d1 = (np.log(S0 / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        bs_price = S0 * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        bs_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")

    # Monte Carlo simulation
    stock_prices = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        Z = np.random.normal(0, 1, num_days - 1)
        path = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z))
        stock_prices[i, 1:] = path

    # Plot some random stock price paths
    plt.figure(figsize=(10, 6))
    plt.title('Simulated Stock Price Paths')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    for i in range(10):  # Plotting 10 random paths for illustration
        plt.plot(np.arange(0, T + dt, dt), stock_prices[i, :])

    plt.show()

    # Calculate the option payoff for each path
    if option_type == 'call':
        option_payoffs = np.maximum(stock_prices[:, -1] - X, 0)
    elif option_type == 'put':
        option_payoffs = np.maximum(X - stock_prices[:, -1], 0)

    # Plot the distribution of option payoffs
    # plt.figure(figsize=(10, 6))
    # plt.hist(option_payoffs, bins=50, color='skyblue', edgecolor='black')
    # plt.title('Distribution of Option Payoffs')
    # plt.xlabel('Option Payoff')
    # plt.ylabel('Frequency')

    # plt.show()

    # Calculate the mean of the discounted payoffs
    option_price = np.mean(option_payoffs * np.exp(-r * T))

    # Combine Black-Scholes estimate and Monte Carlo refinement
    refined_price = 0.5 * (bs_price + option_price)

    # Plot a comparison between the initial Black-Scholes estimate and the refined estimate
    # plt.figure(figsize=(8, 5))
    # plt.bar(['Black-Scholes', 'Refined Monte Carlo'], [bs_price, refined_price], color=['blue', 'green'])
    # plt.title('Option Price Comparison')
    # plt.ylabel('Option Price')
    # plt.show()

    return refined_price

# Example usage
S0 = 100  # Current stock price
X = 100   # Strike price
T = 1     # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility

call_price = black_scholes_monte_carlo(S0, X, T, r, sigma, option_type='call', num_simulations=100000)
# put_price = black_scholes_monte_carlo(S0, X, T, r, sigma, option_type='put', num_simulations=100000)

print(f"Refined Call Option Price: {call_price:.2f}")
# print(f"Refined Put Option Price: {put_price:.2f}")
#10 nota 10