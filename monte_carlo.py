import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class OptionPricingHelper():
    def __init__(self, cur_stock_price, strike_price, days_to_maturity, sigma_volatility, risk_rate, num_simulations):
        self.spot_price = cur_stock_price
        self.strike_price = strike_price
        self.days_to_maturity = days_to_maturity
        self.sigma = sigma_volatility
        self.drift_rate = risk_rate
        self.time_step = 1/365
        self.num_simulations = num_simulations

    def _get_normal_z(self, num_steps):
        """
        Array of normal standard deviations. We can reduce variance by implementing antithetic variates
        by taking half the number of simulations (positive sims) and its negative (negative sims)
        and return two arrays of Z, one positive and one negative.
        """
        Z = np.random.normal(0, 1, size=num_steps)
        
        positive = Z.reshape(-1, 1) # reshape to column vector
        negative = -Z.reshape(-1, 1)

        return positive, negative

    # first step is applying Geometric Brownian Motion
    def brownian_motion_one_path(self):
        Z_pos, Z_neg = self._get_normal_z(self.days_to_maturity - 1)

        price_path_1 = np.zeros(self.days_to_maturity)
        price_path_2 = np.zeros(self.days_to_maturity)

        price_path_1[0] = self.spot_price
        price_path_2[0] = self.spot_price

        drift = (self.drift_rate - 0.5 * self.sigma ** 2) * self.time_step
        diffusion = self.sigma * np.sqrt(self.time_step)

        for t in range(1, self.days_to_maturity):
            price_path_1[t] = price_path_1[t-1] * np.exp(drift + diffusion * Z_pos[t-1, 0])
            price_path_2[t] = price_path_2[t-1] * np.exp(drift + diffusion * Z_neg[t-1, 0])

        return price_path_1, price_path_2

    def brownian_simulations(self):
        """
        Simulate multiple paths of stock prices
        """

        # np.random.seed(3579)

        S1 = np.zeros((self.days_to_maturity, self.num_simulations))
        S2 = np.zeros((self.days_to_maturity, self.num_simulations))

        for each_sim in range(self.num_simulations):

            S1[:, each_sim], S2[:, each_sim] = self.brownian_motion_one_path()

        return S1, S2


    # calculate payoff for calls vs puts
    def option_payoff(self, option_type, S1, S2): # shape (30, 50)
        # underlying stock price at the end
        S1_last_prices = S1[-1, :] # shape (1, 50)
        S2_last_prices = S2[-1, :]
        sum_CT = 0
        sum_CT2 = 0

        # call option
        if option_type == 'call':
            # payoff formula: stock-strike
            S1_diff = S1_last_prices - self.strike_price
            S2_diff = S2_last_prices - self.strike_price
        else:
            # payoff: strike - stock
            S1_diff = self.strike_price - S1_last_prices
            S2_diff = self.strike_price - S2_last_prices

        # max(0, diff)
        S1_max = np.where(S1_diff > 0, S1_diff, 0)
        S2_max = np.where(S2_diff > 0, S2_diff, 0)

        # average of two arrays
        weighted_CT = 0.5 * (S1_max + S2_max)
        sum_CT = np.sum(weighted_CT)
        sum_CT2 = np.sum(weighted_CT**2)

        return sum_CT, sum_CT2

    # discount payoff
    def option_price(self, sum_CT, sum_CT2):
        # discount to present
        discount_factor = np.exp(-self.drift_rate * (self.days_to_maturity / 365))

        # sample means
        E_X = sum_CT / self.num_simulations # E[X]
        E_X2 = sum_CT2 / self.num_simulations # E[X^2]

        # current option price
        cur_price = discount_factor * E_X

        # standard error
        SE = discount_factor * np.sqrt((E_X2 - E_X**2) / self.num_simulations)

        return cur_price, SE


    # visualize simulated paths
    def visualize(self):

        S1, _ = self.brownian_simulations()

        # time steps as x axis
        x_axis = np.linspace(0, self.days_to_maturity, self.days_to_maturity)

        plt.figure(figsize=(8,6))

        for each_path in range(S1.shape[1]): # iterate each path

            plt.plot(x_axis, S1[:, each_path])

        plt.title('Stock Price Simulation')

        plt.show()


class MonteCarloModel:
    def __init__(
        self,
        cur_stock_price: float,
        strike_price: float,
        days_to_maturity: int,
        risk_rate: float,
        sigma_volatility: float,
        num_simulations: int,
        h: float = 1.0,
        ci: float = 0.95,
    ) -> None:
        """
        Monte Carlo pricer with antithetic variates by default.
        Delegates all path‐generation and payoff logic to OptionPricingHelper.
        """
        self.S       = cur_stock_price
        self.K       = strike_price
        self.days    = days_to_maturity
        self.r       = risk_rate
        self.sigma   = sigma_volatility
        self.num_sims= num_simulations
        self.h       = h
        self._ci     = ci
        self.ci_intervals = {}

        # helper always does antithetic internally
        self.option_pricer = OptionPricingHelper(
            cur_stock_price   = self.S,
            strike_price      = self.K,
            days_to_maturity  = self.days,
            sigma_volatility  = self.sigma,
            risk_rate         = self.r,
            num_simulations   = self.num_sims,
        )

        self._S1 = None
        self._S2 = None

    def run_simulation(self) -> None:
        """Generate two antithetic price matrices via the helper."""
        self._S1, self._S2 = self.option_pricer.brownian_simulations()

    def _validate_simulation(self):
        if self._S1 is None or self._S2 is None:
            raise RuntimeError("No simulation found. Call run_simulation() first.")

    def call_option_price(self) -> float:
        self._validate_simulation()
        sum_CT, sum_CT2 = self.option_pricer.option_payoff("call", self._S1, self._S2)
        price, se = self.option_pricer.option_price(sum_CT, sum_CT2)
        z = stats.norm.ppf((1 + self._ci) / 2)
        self.ci_intervals["call"] = z * se
        return price

    def put_option_price(self) -> float:
        self._validate_simulation()
        sum_PT, sum_PT2 = self.option_pricer.option_payoff("put", self._S1, self._S2)
        price, se = self.option_pricer.option_price(sum_PT, sum_PT2)
        z = stats.norm.ppf((1 + self._ci) / 2)
        self.ci_intervals["put"] = z * se
        return price

    def delta(self, option_type: str = "call") -> float:
        """
        Central-difference Delta with consistent random seed
        """
        # Save the current random state
        state = np.random.get_state()
        
        # Set a specific seed for reproducibility across simulations
        np.random.seed(42)
        
        # Base simulation
        self.run_simulation()
        base = self.call_option_price() if option_type == "call" else self.put_option_price()
        
        # Reset seed for identical random numbers
        np.random.seed(42)
        
        # Bump up simulation with same random numbers
        h = 0.01 * self.S  # Use 1% of stock price as step size
        up = OptionPricingHelper(self.S + h, self.K, self.days, self.sigma, self.r, self.num_sims)
        S1_up, S2_up = up.brownian_simulations()
        sum_up, sum2_up = up.option_payoff(option_type, S1_up, S2_up)
        C_up, _ = up.option_price(sum_up, sum2_up)
        
        # Reset seed again
        np.random.seed(42)
        
        # Bump down simulation with same random numbers
        down = OptionPricingHelper(self.S - h, self.K, self.days, self.sigma, self.r, self.num_sims)
        S1_dn, S2_dn = down.brownian_simulations()
        sum_dn, sum2_dn = down.option_payoff(option_type, S1_dn, S2_dn)
        C_dn, _ = down.option_price(sum_dn, sum2_dn)
        
        # Restore the random state
        np.random.set_state(state)
        
        return (C_up - C_dn) / (2 * h)

    def gamma(self, option_type: str = "call") -> float:
        """
        Central-difference Gamma with consistent random seed
        """
        # Save current random state
        state = np.random.get_state()
        
        # Set specific seed
        np.random.seed(42)
        
        # Use 1% of stock price as step size
        h = 0.01 * self.S
        
        # Base simulation
        self.run_simulation()
        base = self.call_option_price() if option_type == "call" else self.put_option_price()
        
        # Reset seed
        np.random.seed(42)
        
        # Up simulation
        up = OptionPricingHelper(self.S + h, self.K, self.days, self.sigma, self.r, self.num_sims)
        S1_up, S2_up = up.brownian_simulations()
        sum_up, sum2_up = up.option_payoff(option_type, S1_up, S2_up)
        C_up, _ = up.option_price(sum_up, sum2_up)
        
        # Reset seed
        np.random.seed(42)
        
        # Down simulation
        down = OptionPricingHelper(self.S - h, self.K, self.days, self.sigma, self.r, self.num_sims)
        S1_dn, S2_dn = down.brownian_simulations()
        sum_dn, sum2_dn = down.option_payoff(option_type, S1_dn, S2_dn)
        C_dn, _ = down.option_price(sum_dn, sum2_dn)
        
        # Restore random state
        np.random.set_state(state)
        
        return (C_up - 2 * base + C_dn) / (h * h)

