import typing as t
import numpy as np
import scipy.stats as stats


class MonteCarloModel:
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        num_sims: int,
        days: int,
        h: float = 1.0,
        ci: float = 0.95,
        antithetic: bool = True,
    ) -> None:
        """
        Initialize model

        Args:
            S (float): Underlying asset price
            K (float): Option strike price
            T (float): Time to expiration in years. Can be provided as days/year, ie.: 180/365
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            num_sims (int): number of simulations to run
            days (int): days to maturity
            h (float): defaults to 1.0. Price bump used in estimating Delta and Gamma
            ci (float): defaults to 0.95. Confidence interval quantile
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.name = "MonteCarlo"

        self.num_sims = num_sims
        self.days_to_mature = days  # N time steps
        self.dt = self.T / self.days_to_mature  # delta t is the time step
        self.h = h
        self.ci_intervals = {}
        self.antithetic = antithetic
        self._simulated_stock_prices = None
        self._simulated_stock_prices_up = None
        self._simulated_stock_prices_down = None
        self._ci_value = ci

    def run_simulation(self) -> None:
        if all(
            [
                self._simulated_stock_prices_down is None,
                self._simulated_stock_prices_up is None,
                self._simulated_stock_prices is None,
            ]
        ):
            self._simulate_price_paths()

    def get_simulated_stock_prices(self) -> np.array:
        return self._simulated_stock_prices

    def call_option_price(self) -> float:
        self._validate_simulation_was_run()
        stock_prices = self.get_simulated_stock_prices()
        price = self._price_option(stock_prices, self.K, self.r, self.T, "call", True)
        return price

    def put_option_price(self) -> float:
        self._validate_simulation_was_run()
        stock_prices = self.get_simulated_stock_prices()
        price = self._price_option(stock_prices, self.K, self.r, self.T, "put", True)
        return price

    def delta(self, option_type: str) -> float:
        """
        Estimate Delta using Finite Difference method

        Returns:
            float: delta
        """
        self._validate_simulation_was_run()
        if option_type == "call":
            delta = self._compute_delta(True)
        if option_type == "put":
            delta = self._compute_delta() - 1
        return delta

    def gamma(self) -> float:
        """
        Estimate Gamma using Finite Difference method

        Returns:
            float: gamma
        """
        self._validate_simulation_was_run()
        gamma = self._compute_gamma(True)
        return gamma

    def simulate_price_paths(self) -> None:
        """
        Simulate all price paths, one with increment `h`, one with decrement `h`, and no increment.
        """
        self._simulate_price_paths()

    def _compute_delta(self, include_ci: bool = False) -> float:
        """
        Estimate Delta and its confidence interval using finite differences. CI is stored in dictionary
        variable `ci_intervals` under key "delta".

        Returns:
            float: estimated average Delta
        """
        discount = np.exp(-self.r * self.T)
        price_up = np.maximum(self._simulated_stock_prices_up - self.K, 0)
        price_down = np.maximum(self._simulated_stock_prices_down - self.K, 0)
        value_up = discount * price_up
        value_down = discount * price_down
        deltas = (value_up - value_down) / (2 * self.h)
        if include_ci:
            ci = self._compute_ci(deltas, self._ci_value)
            self.ci_intervals.update({"delta": ci})
        return np.mean(deltas)

    def _compute_gamma(self, include_ci: bool = False) -> float:
        """
        Estimate Gamma and its confidence interval using finite differences. CI is stored in dictionary
        variable `ci_intervals` under key "gamma".

        Returns:
            float: estimated average Gamma
        """
        discount = np.exp(-self.r * self.T)
        price_up = np.maximum(self._simulated_stock_prices_up - self.K, 0)
        price_down = np.maximum(self._simulated_stock_prices_down - self.K, 0)
        price = np.maximum(self._simulated_stock_prices - self.K, 0)
        value_up = discount * price_up
        value_down = discount * price_down
        discounted = discount * price
        gammas = (value_up - 2 * discounted + value_down) / (self.h**2)
        if include_ci:
            ci = self._compute_ci(gammas, self._ci_value)
            self.ci_intervals.update({"gamma": ci})
        return np.mean(gammas)

    @staticmethod
    def _compute_ci(array: np.array, ci: float) -> float:
        """
        Compute confidence interval given an array of option prices.
        """
        se = np.std(array, ddof=1) / np.sqrt(array.shape[0])
        z = stats.norm.ppf((1 + ci) / 2)
        return z * se

    @staticmethod
    def _get_normal_z(num_sims: int, days: int, antithetic: bool = False) -> np.array:
        """
        Array of normal standard deviations. When antithetic is true then we half the number of simulations
        (positive sims) and we concatenate negative simulations.
        """
        half_sims = num_sims // 2 if antithetic else num_sims
        Z = np.random.standard_normal((days - 1, half_sims))
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=1)
        return Z

    def _price_option(
        self, stock_prices: np.array, K: float, r: float, T: float, option_type: str = "call", include_ci: bool = False
    ) -> float:
        """
        Price a call or put option using simulated stock prices.

        Returns:
            float: option value
        """
        discount = np.exp(-r * T)
        if option_type == "call":
            option_price = np.maximum(stock_prices - K, 0)
        if option_type == "put":
            option_price = np.maximum(K - stock_prices, 0)
        if include_ci:
            discounted = discount * option_price
            ci = self._compute_ci(discounted, self._ci_value)
            self.ci_intervals.update({option_type: ci})
        return discount * np.mean(option_price)

    def _simulate_prices(
        self, S: float, T: float, r: float, sigma: float, num_sims: float, days: float, h: float, antithetic: bool
    ) -> t.Tuple[np.array, np.array, np.array]:
        """
        Simulate stock prices using Brownian motion. Note, we simulate three price paths using same
        Z (standard normal) to later estimate Delta and Gamma. Using the same Z ensures stability
        in Delta and Gamma.

        Returns:
            tuple: tuple of arrays containing simulated price without increment, w/increment, and w/decrement
        """
        Z = self._get_normal_z(num_sims, days, antithetic)
        simulated_prices = self._simulate_single_price_path(S, T, r, sigma, days, Z)
        simulated_prices_up = self._simulate_single_price_path(S + h, T, r, sigma, days, Z)
        simulated_prices_down = self._simulate_single_price_path(S - h, T, r, sigma, days, Z)
        return simulated_prices, simulated_prices_up, simulated_prices_down

    @staticmethod
    def _simulate_single_price_path(S: float, T: float, r: float, sigma: float, days: float, Z: np.array) -> np.array:
        """
        Simulate a single price path using Brownian motion. Prices are initialized at S and then simulated
        for each day until expiration.

        Returns:
            np.array: array of simulated prices
        """
        prices = np.zeros((days, Z.shape[1]))
        prices[0] = S
        dt = T / days
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        for t in range(1, days):
            prices[t] = prices[t - 1] * np.exp(drift + vol * Z[t - 1])
        # take last day prices at expiration
        return prices[-1]

    def _simulate_price_paths(self) -> None:
        """
        Simulate stock price paths with increment, decrement, and at initial price S
        """

        self._simulated_stock_prices, self._simulated_stock_prices_up, self._simulated_stock_prices_down = (
            self._simulate_prices(
                self.S, self.T, self.r, self.sigma, self.num_sims, self.days_to_mature, self.h, self.antithetic
            )
        )

    def _validate_simulation_was_run(self) -> None:
        if all(
            [
                self._simulated_stock_prices_down is None,
                self._simulated_stock_prices_up is None,
                self._simulated_stock_prices is None,
            ]
        ):
            raise RuntimeError("No simulated prices available. Please call `run_simulation` method first")
