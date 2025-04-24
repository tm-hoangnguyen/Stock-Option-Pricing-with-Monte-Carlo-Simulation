"""
Reference Source:
https://theaiquant.medium.com/mastering-the-black-scholes-model-with-python-a-comprehensive-guide-to-option-pricing-11af712697b7

Black Scholes Model is mathematical technique, which will be used used to benchmark option prices,
including call, put, delta, and gamma, against results from Monte Carlo simulation.
"""

import numpy as np
import scipy.stats as stats

class BlackScholesOption:
    def __init__(self, spot_p: float, strk_P: float, exp_t: float, int_r: float, vol: float) -> None:
        self.spot_p = spot_p  # Current price of the underlying asset
        self.strk_P = strk_P  # The strike price of the option
        self.exp_t = exp_t/365  # Time to expiration in years (can be in fraction)
        self.int_r = int_r  # The risk-free rate
        self.vol = vol  # The volatility of the underlying asset

    def _calculate_d1_and_d2(self) -> tuple:
        "This is the helper method to calculate d1 and d2 used in option pricing formulas."
        log_price_ratio = np.log(self.spot_p / self.strk_P)
        sigma_sq = 0.5 * self.vol ** 2

        # d1 = (ln(S / K) + (r + 1/2 * σ^2) * T) / (σ * sqrt(T)) -- stdev of stock price above strike price
        d1 = (log_price_ratio + (self.int_r + sigma_sq) * self.exp_t) / (self.vol * np.sqrt(self.exp_t))

        # d2 = d1 - σ * sqrt(T) -- d2 derived from d1
        d2 = d1 - self.vol * np.sqrt(self.exp_t)
        return d1, d2

    def get_option_metrics(self) -> dict:
        "The following code calculate the call price, put price, delta for both call and put, and gamma in one method."
        "It then returns all these values in a dictionary."
        
        d1, d2 = self._calculate_d1_and_d2()

        # Getting call and Put prices. Also getting delta and gamma
        call_price = self.spot_p * stats.norm.cdf(d1) - self.strk_P * np.exp(-self.int_r * self.exp_t) * stats.norm.cdf(d2)
        put_price = self.strk_P * np.exp(-self.int_r * self.exp_t) * stats.norm.cdf(-d2) - self.spot_p * stats.norm.cdf(-d1)

        delta_call = stats.norm.cdf(d1)
        delta_put = -stats.norm.cdf(-d1)


        gamma_value = stats.norm.pdf(d1) / (self.spot_p * self.vol * np.sqrt(self.exp_t))

        # Returning all metrics as a dictionary
        return {
            "call_price": call_price,
            "put_price": put_price,
            "delta_call": delta_call,
            "delta_put": delta_put,
            "gamma": gamma_value
        }


# # example usage
# black_scholes = BlackScholesOption(100, 120, 1, 0.05, 0.2)

# # call option price
# print(black_scholes.get_option_metrics()['gamma'])
