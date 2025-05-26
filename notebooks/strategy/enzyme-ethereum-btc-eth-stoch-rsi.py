"""ETH/BTC stochastic trading strategy.

To backtest this strategy module using Docker compose:

.. code-block:: console

    source scripts/set-latest-tag-gcp.sh
    docker-compose run ethereum-btc-eth-stoch-rsi backtest

Or locally in the dev environment:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategy/enzyme-ethereum-btc-eth-stoch-rsi.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

"""

import datetime
import logging

import pandas_ta


from tradingstrategy.chain import ChainId
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradingstrategy.client import Client
from tradeexecutor.utils.binance import create_binance_universe
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.lending import LendingProtocolType, LendingReserveDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.utils.crossover import contains_cross_over, contains_cross_under
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.pandas_trader.strategy_input import _calculate_and_cache_candle_width


trading_strategy_engine_version = "0.5"


class Parameters:

    id = "enzyme-ethereum-btc-eth-stoch-rsi" # Used in cache paths

    cycle_duration = CycleDuration.cycle_7d
    candle_time_bucket = TimeBucket.d7
    credit_allocation = 1.0
    rsi_length = 26
    stoch_rsi_low = 20
    stoch_rsi_high = 40 
    stoch_rsi_length = 19

    # stop_loss_pct = Real(0.7, 0.99)
    stop_loss_pct = 0.9
    trailing_stop_loss_pct = 0.80 
    trailing_stop_loss_activation_level = 1.0 

    #
    # Live trading only
    #
    chain_id = ChainId.ethereum
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(weeks=rsi_length*2 + stoch_rsi_length + 2)  # see pandas_ta.stoch_rsi for how much data is needed
    trading_strategy_engine_version = "0.5"
    
    #
    # Backtesting only
    #

    # Use Binance data in backtesting,
    # We get a longer, more meaningful, history but no credit simulation.
    binance_data = False

    if binance_data:
        backtest_start = datetime.datetime(2020, 1, 1)
        # backtest_end = datetime.datetime(2024, 4, 20)
        # backtest_start = datetime.datetime(2022, 6, 1)
        backtest_end = datetime.datetime(2024, 7, 15)

    else:
        # dex dates
        backtest_start = datetime.datetime(2021, 4, 1)
        backtest_end = datetime.datetime(2024, 5, 15)
    
    stop_loss_time_bucket = TimeBucket.h4
    backtest_trading_fee = 0.0005
    initial_cash = 10_000



def get_strategy_trading_pairs(mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Get trading pairs the strategy uses
    
    - Different options for backtesting
    """
    use_binance = mode.is_backtesting() and Parameters.binance_data 

    if use_binance:
        trading_pairs = [
            (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
            (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]
    else:
        trading_pairs = [
            (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.0030),  # Deep liquidity
            (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),  # Deep liquidity
            (ChainId.ethereum, "uniswap-v3", "SOL", "WETH", 0.003)  # Deep liquidity
        ]

    return trading_pairs


def get_lending_reserves(mode: ExecutionMode) -> list[LendingReserveDescription]:
    """Get lending reserves the strategy needs."""
    
    use_binance = mode.is_backtesting() and Parameters.binance_data 

    if use_binance:
        # Credit interest is not available on Binance
       return []
    else:
        # We use Aave v2 in backtesting (longer history)
        # and Aave v3 in live execution (more liquid market)
        if mode.is_backtesting():
            lending_reserves = [
               (ChainId.ethereum, LendingProtocolType.aave_v2, "USDC"),
            ]
        else:
            lending_reserves = [
               (ChainId.ethereum, LendingProtocolType.aave_v3, "USDC"),
            ]

    return lending_reserves


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - In this example, we load all Binance spot data based on our Binance trading pair list.
    """
    trading_pairs = get_strategy_trading_pairs(execution_context.mode)
    lending_reserves = get_lending_reserves(execution_context.mode)

    use_binance = trading_pairs[0][0] == ChainId.centralised_exchange

    if use_binance:
        # Backtesting - load Binance data
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in trading_pairs],
            candle_time_bucket=Parameters.candle_time_bucket,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            start_at=universe_options.start_at,
            end_at=universe_options.end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
            include_lending=False,
            forward_fill=True,
        )
    else:

        if execution_context.live_trading or execution_context.mode == ExecutionMode.preflight_check:
            start_at, end_at = None, None
            required_history_period=Parameters.required_history_period
        else:
            required_history_period = None
            start_at=universe_options.start_at
            end_at=universe_options.end_at

    
        dataset = load_partial_data(
            client,
            execution_context=execution_context,
            time_bucket=Parameters.candle_time_bucket,
            pairs=trading_pairs,
            universe_options=universe_options,
            start_at=start_at,
            end_at=end_at,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            lending_reserves=lending_reserves,
            required_history_period=required_history_period
        )

        # Filter down to the single pair we are interested in
        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            forward_fill=True,
        )
    return strategy_universe


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()

    indicators.add(
        "stoch_rsi",
        pandas_ta.stochrsi,
        {"length": parameters.stoch_rsi_length, 'rsi_length': parameters.stoch_rsi_length, 'k': 3, 'd': 3},  # No parameters needed for this custom function
        IndicatorSource.close_price,
    )

    indicators.add(
        "rsi",
        pandas_ta.rsi,
        {"length": parameters.rsi_length},
        IndicatorSource.close_price,
    )

    return indicators




def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    # 
    # Decidion cycle setup.
    # Read all variables we are going to use for the decisions.
    #
    parameters: Parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe
    cash = position_manager.get_current_cash()
    trading_pairs = get_strategy_trading_pairs(input.execution_context.mode)
    lending_reserves = get_lending_reserves(input.execution_context.mode)

    trades = []

    # Enable trailing stop loss after we reach the profit taking level
    #
    for position in state.portfolio.open_positions.values():
        if not position.is_credit_supply():
            if position.trailing_stop_loss_pct is None:
                close_price = indicators.get_price(position.pair)
                if close_price >= position.get_opening_price() * parameters.trailing_stop_loss_activation_level:
                    position.trailing_stop_loss_pct = parameters.trailing_stop_loss_pct 

    # Setup asset allocation parameters
    max_assets = len(trading_pairs)
    allocation = round(1/max_assets - 0.01, 2)
    use_credit = len(lending_reserves) > 0

    # If any of trading pairs enters to long position,
    # close our credit position
    credit_closed = False
    traded_this_cycle = False
    available_cash = cash
    ready = False

    for pair_desc in trading_pairs:
        
        #
        # Indicators
        #
        pair = strategy_universe.get_pair_by_human_description(pair_desc)

        close_price = indicators.get_price(pair=pair)  # Price the previous 15m candle closed for this decision cycle timestamp
        rsi_k = indicators.get_indicator_value("stoch_rsi", pair=pair, column=f'STOCHRSIk_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  
        rsi_d = indicators.get_indicator_value("stoch_rsi", pair=pair, column=f'STOCHRSId_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  

        # Visualisations
        #
        if input.is_visualisation_enabled():
            visualisation = state.visualisation
            visualisation.plot_indicator(timestamp, f"RSI Stochastic {pair.base}", PlotKind.technical_indicator_detached, rsi_d, pair=pair)
            visualisation.plot_indicator(timestamp,f"rsi_k {pair}", PlotKind.technical_indicator_overlay_on_detached, rsi_k, pair=pair, detached_overlay_name=f"RSI Stochastic {pair.base}")
            visualisation.plot_indicator(timestamp,f"Rsi Stochastic Low {pair}", PlotKind.technical_indicator_overlay_on_detached, parameters.stoch_rsi_low, pair=pair, detached_overlay_name=f"RSI Stochastic {pair.base}")

        if None in (rsi_k, rsi_d, close_price):
            # Not enough historic data,
            # cannot make decisions yet.
            # Should never happen in live trading,
            # so try to print out some useful diagnostics what might
            # be wrong.
            if input.execution_context.mode.is_live_trading():
                stoch_rsi_df = indicators.get_indicator_dataframe("stoch_rsi", pair=pair)
                candle_width = _calculate_and_cache_candle_width(stoch_rsi_df.index)
                position_manager.log(
                    f"Strategy does not have enough data to make any decisions for the pair {pair}\n" +
                    f"close_price: {close_price}\n" + 
                    f"rsi_k: {rsi_k}\n" +
                    f"rsi_d: {rsi_d}\n" +
                    f"stoch_rsi columns: {stoch_rsi_df.columns}\n" +
                    f"stoch_rsi index: {stoch_rsi_df.index}\n" +
                    f"stoch_rsi_length: {parameters.stoch_rsi_length}\n" +
                    f"dataframe candle width: {candle_width}",
                    level=logging.WARNING,
                )
                continue

        rsi_k_series = indicators.get_indicator_series("stoch_rsi", pair=pair, column=f'STOCHRSIk_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  
        rsi_d_series = indicators.get_indicator_series("stoch_rsi", pair=pair, column=f'STOCHRSId_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  

        if len(rsi_k_series) < 2:
            continue

        ready = True

        crossover, crossover_index = contains_cross_over(
                rsi_k_series,
                rsi_d_series,
                lookback_period=2,
                must_return_index=True
            )

        crossunder, crossunder_index = contains_cross_under(
                rsi_k_series,
                rsi_d_series,
                lookback_period=2,
                must_return_index=True
            )
        #
        # Trading logic
        #
        if crossover and crossover_index == -1 :
            if len(state.portfolio.open_positions) >= max_assets:
                pass
                # print(f"Want to place in a trade but there are already {len(state.portfolio.open_positions)} positions open and max is {parameters.max_assets} for pair {pair.base}")

        # Check for open condition - is the price breaking out
        #
        non_credit_open_positions = [p for p in state.portfolio.open_positions.values() if not p.is_credit_supply()]
        if len(non_credit_open_positions) < max_assets and state.portfolio.get_open_position_for_pair(pair) is None:        
            if  crossover and crossover_index == -1 :
                # close credit supply position before opening a new long position
                if position_manager.is_any_credit_supply_position_open():
                    #print(f"Closing credit supply position on {timestamp}")
                    if not credit_closed:
                        current_pos = position_manager.get_current_credit_supply_position()
                        new_trades = position_manager.close_credit_supply_position(current_pos)
                        trades += new_trades
                        # Est. available cash after all credit positions are closed
                        available_cash += float(current_pos.get_quantity()) 
                        credit_closed = True

                trades += position_manager.open_spot(
                    pair,
                    value=available_cash * allocation,
                    stop_loss_pct=parameters.stop_loss_pct,             
                )

                traded_this_cycle = True
        else:
            # Check for close condition
            if  crossunder and crossunder_index == -1 and rsi_d > parameters.stoch_rsi_high and state.portfolio.get_open_position_for_pair(pair) is not None:
                position = state.portfolio.get_open_position_for_pair(pair)
                trades += position_manager.close_position(position)
                traded_this_cycle = True

    # We have accumulatd enough data to make the first real (non credit) trading decision.
    # This allows us to have fair buy-and-hold vs backtest period comparison
    if ready:
        state.mark_ready(timestamp)

    # If we have any access cash or new deposit, move them to Aave
    if use_credit and not traded_this_cycle:
        cash_to_deposit = available_cash * 0.99
        new_trades = position_manager.add_cash_to_credit_supply(cash_to_deposit)
        trades += new_trades
    
    return trades


#
# Strategy metadata.
#
# Displayed in the user interface.
#

sort_priority = -1

tags = {StrategyTag.beta, StrategyTag.live}

name = "Stochastic ETH/BTC long"

short_description = "A breakout strategy for ETH and BTC using Stochastic RSI indicators"

icon = "https://tradingstrategy.ai/avatars/arbitrum-stoch-rsi.webp"

long_description = """
# Strategy description

This strategy leverages the Stochastic RSI indicator to identify long-only opportunities on multiple trading pairs on the Ethereum blockchain, specifically ETH and BTC.

- Trades on WBTC/USDC and WETH/USDC on Uniswap V3.
- Designed to capture long-term trends while minimizing drawdowns.
- The strategy focuses on weekly cycles, rebalancing every 7 days.
- It performs well in trending markets and aims to protect capital during downturns with stop-loss mechanisms.
- During the bear and crab markets excess cash is deposited to Aave v3 as USDC to gain credit supply yield.

**Past performance is not indicative of future results**.

## Assets and trading venues

- The strategy trades on decentralized exchanges (DEX) such as Uniswap V3 on the Ethereum blockchain.
- Trading pairs include WBTC/USDC and WETH/USDC with the fee tier of 30 bps and 5 bps
- Keeps reserves in stablecoins such as USDC, and deposits to Aave USDC reserves for interest.>>>>>>> e0ba15c419ce942e122d9e389594787eca5ec556

## Stochastic trading

This strategy is built around Stochastic RSI indicator. It uses this indicator to detect breakout conditions
in the underlying trading pairs and allocate the strategy equity to a breakout trade.

## Backtesting

The backtesting was performed using data from Binance.

- Binance data was used, as Uniswap V3 does not have enough trading history to give meaningful backtest results.
- Backtesting does not include the interest gained from Aave USDC deposits
- [See backtesting results](./backtest)
- [Read more about backtesting](https://tradingstrategy.ai/glossary/backtest).

## Risk

The strategy has a backtested maximum drawdown of **-27%**. It employs strict stop-loss and trailing stop mechanisms to mitigate losses.

For further understanding the key aspects of risks:
- The strategy does not use any leverage.
- Trades only highly liquid pairs to ensure minimal slippage and robust trade execution.
- Decentralised finance is very novel and there is high risk of ruin if any of the underlying DeFi protocols get hacked.

## Benchmark

Here are some benchmarks comparing the strategy's performance with other indices.

|                              | CAGR | Maximum drawdown | Sharpe |
|------------------------------|------|------------------|--------|
| This strategy                | 69%  | -27%             | 1.60   |
| SP500 (20 years)             | 11%  | -33%             | 0.72   |
| Bitcoin (backtesting period) | 50%  | -76%             | 0.98   |
| Ether (backtesting period)   | 90%  | -80%             | 1.22   |

Sources:

- [Our strategy](./backtest)
- [Buy and hold BTC](./backtest)
- [Buy and hold ETH](./backtest)
- [What is CAGR - Compound annual growth rate](https://tradingstrategy.ai/glossary/compound-annual-growth-rate-cagr)
- [What is maximum drawdown](https://tradingstrategy.ai/glossary/maximum-drawdown)
- [What is Sharpe ratio](https://tradingstrategy.ai/glossary/sharpe)
- [SP500 stock index](https://curvo.eu/backtest/en/portfolio/s-p-500--NoIgygZACgBArABgSANMUBJAokgQnXAWQCUEAOAdlQEYBdeoA?config=%7B%22periodStart%22%3A%222004-02%22%7D)

## Trading frequency

The strategy operates on a weekly cycle, rebalancing every Monday and adjusting positions as necessary based on Stochastic RSI signals.

## Robustness

This strategy was tested on Binance dataset. 

- Uniswap price behavior and Binance price behavior should be almost equal due to arbitrage on the large trading pairs.
- The strategy does not have extensive robustness analysis, but is more based on the general idea of the trade.
- As crypto markets are new, there is unlikely to be enough market data to have a statistically significant results. 
- There was no parameter sensitivity analysis done.

## Updates

This strategy is periodically reviewed and updated to incorporate the latest market data and trading techniques. Stay tuned for updates via the [Trading Strategy community](https://tradingstrategy.ai/community).

## Further information

- Any questions are welcome in [the Discord community chat](https://tradingstrategy.ai/community).
- See the blog post [on how this strategy is constructed](https://tradingstrategy.ai/blog/outperfoming-eth) for more details.

"""
