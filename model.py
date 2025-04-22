import typing as t
import black_scholes as bs
import monte_carlo as mc


class OptionsPricer:
    """
    Wrapper around BlackScholes and MonteCarlo models.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        num_sims: int = 1000,
        days: int = 30,
        h: float = 1.0,
        ci: float = 0.95,
        antithetic: bool = False,
    ) -> None:
        """
        Initialize model

        Args:
            S (float): Underlying asset price
            K (float): Option strike price
            T (float): Time to expiration in years. Can be provided as days/year, ie.: 180/365
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            num_sims (int): number of Monte Carlo simulations to run, defaults to 1000
            days (int): days until maturity, defaults to 30
            h (float): Price bump used in estimating Delta and Gamma, defaults to 1.0
            ci (float): defaults to 0.95. Confidence interval quantile
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.black_scholes = bs.BlackScholesOption(self.S, self.K, self.T, self.r, self.sigma)
        self.monte_carlo = mc.MonteCarloModel(
            self.S, self.K, self.T, self.r, self.sigma, num_sims, days, h, ci, antithetic
        )

    def price_options(self) -> None:

        model = self.black_scholes

        print("")
        print("*** BlackScholes Results ***")
        print("-" * 50)
        print(f"Call Option Price: {round(model.get_option_metrics()['call_price'], 2)}")
        print(f" Put Option Price: {round(model.get_option_metrics()['put_price'], 2)}")
        print(f"       Delta Call: {round(model.get_option_metrics()['delta_call'], 4)}")
        print(f"        Delta Put: {round(model.get_option_metrics()['delta_put'], 4)}")
        print(f"            Gamma: {round(model.get_option_metrics()['gamma'], 4)}")

        print("\n *** Monte Carlo Results ***")
        mc_model = self.monte_carlo
        mc_model.run_simulation()
        call_price = mc_model.call_option_price()
        put_price = mc_model.put_option_price()
        delta_call = mc_model.delta("call")
        delta_put = mc_model.delta("put")
        gamma = mc_model.gamma()

        print("-" * 50)
        print(f"Call Option Price: {round(call_price, 2)} (+/- {round(mc_model.ci_intervals['call'], 4)})")
        print(f" Put Option Price: {round(put_price, 2)}  (+/- {round(mc_model.ci_intervals['put'], 4)})")
        print(f"       Delta call: {round(delta_call, 4)}  (+/- {round(mc_model.ci_intervals['delta'], 4)})")
        print(f"        Delta put: {round(delta_put, 4)}  (+/- {round(mc_model.ci_intervals['delta'], 4)})")
        print(f"            Gamma: {round(gamma, 4)}  (+/- {round(mc_model.ci_intervals['gamma'], 4)})")
        print("-" * 50)