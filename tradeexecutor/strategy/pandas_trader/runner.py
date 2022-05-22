"""A strategy runner that does simple execution of trades based on Pandas calculations."""

import datetime
from typing import List, Optional, Dict
import logging

import pandas as pd

from tradeexecutor.strategy.pandas_trader.brain import StrategyBrain
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed


logger = logging.getLogger(__name__)


class PandasTraderRunner(StrategyRunner):
    """A trading executor for Pandas math based algorithm."""

    def __init__(self, *args, brain: StrategyBrain, max_data_age: Optional[datetime.timedelta] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = brain
        self.max_data_age = max_data_age

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, executor_universe: TradingStrategyUniverse, state: State, debug_details: dict) -> List[TradeExecution]:
        """Run one strategy tick."""

        assert isinstance(executor_universe, TradingStrategyUniverse)
        universe = executor_universe.universe
        pd_timestamp = pd.Timestamp(clock)
        return self.brain(
            timestamp=pd_timestamp,
            universe=universe,
            state=state,
            debug_details=debug_details,
        )

    def pretick_check(self, ts: datetime.datetime, universe: TradingStrategyUniverse):
        """Check the data looks more or less sane."""

        assert isinstance(universe, TradingStrategyUniverse)
        universe = universe.universe

        now_ = ts

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        # Don't assume we have candle or liquidity data e.g. for the testing strategies
        if universe.candles.get_candle_count() > 0:
            start, end = universe.get_candle_availability()

            if self.max_data_age is not None:
                if now_ - end > self.max_data_age:
                    raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")