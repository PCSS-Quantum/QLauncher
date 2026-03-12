import itertools
from collections.abc import Sequence

import numpy as np
import yfinance as yf

from qlauncher.base import Problem
from qlauncher.base.problem_like import FN


class PortfolioOptimization(Problem):
	def __init__(
		self,
		tickers: Sequence[str] = None,
		budget: int = None,
		target_volatility: float = None,
		period_start: str = '2020-01-01',
		period_end: str = '2025-01-01',
		days_invested=252,
		instance_name: str = 'unnamed',
	):
		self.tickers = tickers
		self.budget = budget if budget else len(tickers) // 2
		self.target_volatility = target_volatility
		self.days_invested = days_invested

		data = yf.download(self.tickers, start=period_start, end=period_end)['Close']

		returns = data.pct_change().dropna()

		self.mean_daily_returns = returns.mean()

		self.cov_matrix = returns.cov()

		super().__init__((self.mean_daily_returns, self.cov_matrix, self.budget, self.target_volatility, days_invested), instance_name)

	def to_fn(self, budget_penalty_weight, volatility_penalty_weight) -> FN:
		def cost_fn(bit_string) -> float:
			budget_penalty = abs(sum(bit_string) - self.budget) * budget_penalty_weight

			portfolio_return = np.sum(self.mean_returns * bit_string) * self.days_invested

			volatility = np.sqrt(np.dot(bit_string.T, np.dot(self.cov_matrix, bit_string)))
			volatility_penalty = max(0, volatility - self.target_volatility) * volatility_penalty_weight

			return budget_penalty + volatility_penalty - portfolio_return

		return FN(cost_fn)

	def get_extreme_volatilities(self) -> tuple:
		"""Returns: average volatility, maximum volatility and minimum volatility. Uses full space search."""

		n = len(self.tickers)

		bitstrings = np.array(list(itertools.product([0, 1], repeat=n)))

		variances = np.einsum('ij,jk,ik->i', bitstrings, self.cov_matrix, bitstrings)

		volatilities = np.sqrt(variances)

		valid_volatilities = volatilities[1:]

		return (np.mean(valid_volatilities), np.max(valid_volatilities), np.min(valid_volatilities))
