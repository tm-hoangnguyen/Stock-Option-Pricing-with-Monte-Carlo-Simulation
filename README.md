### This project simulates European option pricing using Monte Carlo methods, incorporating Delta and Gamma estimation, as well as variance reduction through antithetic variates.

## Files

The project contains the following files:

- `monte_carlo.py`  
  Implements the Monte Carlo simulation logic. This is the core of our project, it also includes variance reduction and CI analysis.

- `black_scholes.py`  
  Implements the Black-Scholes formulas for pricing European call and put options. Uses as a benchmark for the simulation models.

- `model.py`  
  A wrapper module that integrates functions from `black_scholes.py` and `monte_carlo.py`.

- `demo.ipynb`  
  A Jupyter Notebook demonstrating how to use the simulation. It benchmarks the results against the Black-Scholes model and fetches market data using the **yfinance** API.
