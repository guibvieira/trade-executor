"""An example strategy that does a random buy or sell every day using WETH-USDC pair."""
import logging
import random
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel

from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.universe_model import StaticUniverseModel
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


# Cannot use Python __name__ here because the module is dynamically loaded
logger = logging.getLogger("uniswap_simulatead_example")


class DummyAlphaModel(AlphaModel):
    """Hold random % of portfolio in ETH-USDC, random in cash."""

    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[int, float]:

        # Because this is a test strategy, we assume we have fixed 3 assets
        assert len(universe.exchanges) == 1
        uniswap = universe.get_single_exchange()
        assert universe.pairs.get_count() == 1

        weth_usdc = universe.pairs.get_one_pair_from_pandas_universe(uniswap.exchange_id, "WETH", "USDC")
        alphas = {weth_usdc.pair_id: random.random()}
        logger.info("Our alphas are %s", alphas)
        return alphas


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModel,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        revaluation_method: RevaluationMethod,
        client,
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        universe_model: StaticUniverseModel,
        **kwargs) -> StrategyExecutionDescription:

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    # Use static universe passed from the tests
    assert isinstance(universe_model, StaticUniverseModel)

    runner = QSTraderRunner(
        alpha_model=DummyAlphaModel(),
        timed_task_context_manager=timed_task_context_manager,
        execution_model=execution_model,
        approval_model=approval_model,
        revaluation_method=revaluation_method,
        sync_method=sync_method,
        pricing_model_factory=pricing_model_factory,
        cash_buffer=0.5,
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]