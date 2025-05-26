#!/usr/bin/env python
# coding: utf-8

# # Alternative data example
# 
# An example strategy backtest how to load your own signal data from CSV.
# 
# - Based on `portfolio-construction` example
# - Loads a custom data feed ("alternative data") from for multiple trading pairs 
#     - Loads a CSV file
#     - Maps CSV data to trading pairs based on a base token smart contract address
#     - Creates different indicators based on the CSV data
# - Check that the alternative data for signal looks good
# - Draws an alternative visualisation for a single pair for diagnostics purposes
# - Run a [portfolio construction strategy](https://tradingstrategy.ai/blog/writing-portfolio-construction-strategy-in-python) based on the signal
#     - In a real strategy this signal needs to be combined with other signals.
# 
# 
# **Note**: This is a limited data sample. In this example, we have good match only for two tokens/trading pairs AAVE and MATIC. Other trading pairs do not have either trades (price data) or good signal data.

# # Set up
# 
# Set up Trading Strategy data client.
# 

# In[1]:


from tradeexecutor.utils.notebook import setup_charting_and_output
from tradingstrategy.client import Client
from tradeexecutor.utils.notebook import setup_charting_and_output, OutputMode
import ipdb

client = Client.create_jupyter_client()

# Set up drawing charts in interactive vector output mode.
# This is slower. See the alternative commented option below.
# Kernel restart needed if you change output mode.
# setup_charting_and_output(OutputMode.interactive)

# Set up rendering static PNG images.
# This is much faster but disables zoom on any chart.
setup_charting_and_output(OutputMode.static, image_format="png", width=1500, height=1000)


# # Prerequisites
# 
# To run this backtest, you first need to run `scripts/prefilter-arb.py` to build a Arbitrum dataset (> 1 GB) for this backtest based on more than 10 GB downloaded data.

# In[36]:


from pathlib import Path
import pandas as pd
# See scripts/prefilter-polygon.py
liquidity_output_fname = Path("/tmp/arb-liquidity-prefiltered.parquet")
price_output_fname = Path("/tmp/arb-price-prefiltered.parquet")

assert price_output_fname.exists(), "Run prefilter script first"
assert liquidity_output_fname.exists(), "Run prefilter script first"


# # Custom data

# Load the custom data from a CSV file.
# 
# - Load using Pandas
# - This data will be split and mapped to per-pair indicators later on, as the data format is per-pair
# 
# *Note*: Relative paths work different in different notebook run-time environments. Below is for Visual Studio Code.

# In[37]:


import os
import pandas as pd

NUM_TOKENS = 15

chain_name = 'arbitrum'

#Import Alerted Coins
df_alerts_arbitrum = pd.read_csv(f'gs://taraxa-research/trading_strategies_ai/{chain_name}_alerts.csv')
df_alerts_arbitrum = df_alerts_arbitrum[df_alerts_arbitrum['total_volume'] > 10000]
df_alerts_arbitrum = df_alerts_arbitrum.drop_duplicates(subset=['coin_id', 'symbol', 'name'], keep='last')[['coin_id', 'symbol', 'name', 'contract_address', 'platform_name', 'market_cap', 'total_volume']]

# custom_data_df = pd.read_csv(CSV_PATH)  # For the repo, we keep a partial sample of the data
custom_data_df = pd.read_parquet(f'gs://taraxa-research/trading_strategies_ai/df_trend_arbitrum_lc_sent.parquet')

custom_data_df.index = pd.DatetimeIndex(custom_data_df["date"])
custom_data_df["contract_address"] = custom_data_df["contract_address"].str.lower()  # Framework operates with lowercased addresses everywehre

start = custom_data_df.index[0]
end = custom_data_df.index[-1]

csv_token_list = list(custom_data_df.contract_address.unique())
print(f"CSV contains data for {len(csv_token_list)} tokens, time range {start} - {end}")

csv_token_list_backtest = df_alerts_arbitrum.sort_values(by='total_volume', ascending=False).iloc[:NUM_TOKENS]
print(f"Pre-selecting the following tokens and contract addresses to backtest {csv_token_list_backtest}")

arb_erc20_address_list = csv_token_list_backtest['contract_address'].tolist()
arb_erc20_address_list = [address.lower() for address in set(arb_erc20_address_list)]


# Create per-pair DataFrame group by
custom_data_df.rename(columns={'coin_id_count': "social_mentions"}, inplace=True)
custom_data_group = custom_data_df.groupby("contract_address")


# # Parameters
# 
# - Strategy parameters define the fixed and grid searched parameters

# In[38]:


from itertools import combinations

# Define your single conditions
single_conditions = [
    'social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 
    'posts_created_increasing', 'posts_active_increasing', 'social_dominance_increasing', 
    'contributors_active_increasing', 'contributors_created_increasing',
    'social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 
    'posts_created_ema_cross_over', 'posts_active_ema_cross_over', 'social_dominance_ema_cross_over', 
    'contributors_active_ema_cross_over', 'contributors_created_ema_cross_over'
]

# Generate all combinations of up to 3 conditions
trigger_conditions = [list(combo) for r in range(1, 4) for combo in combinations(single_conditions, r)]

# Encode each combination
encoded_conditions = ['&'.join(cond) for cond in trigger_conditions]


# In[39]:


from tradingstrategy.chain import ChainId
import datetime

from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.reserve_currency import ReserveCurrency


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "portfolio-construction" # Used in cache paths

    cycle_duration = CycleDuration.d1  # Daily rebalance
    candle_time_bucket = TimeBucket.d1  
    allocation = 0.085   
    
    max_assets = 10  # How many assets hold in portfolio once
    rsi_length = 7   # Use 7 days RSI as the alpha signal for a trading pair

    adx_length = 14  # 14 days
    adx_filter_threshold = 25

    #
    # Liquidity risk analysis and data quality
    #
    min_liquidity = 10000  # Do not trade pairs below this amount of USD locked in TVL
    min_price = 0.00000001  # Filter out trading pairs with bad price units
    max_price = 1_000_000  # Filter out trading pairs with bad price units

    #Trigger
    bb_length = 40
    bb_std = 2

    #Safety Guards
    sma_length = 20
    min_volume = 5000
    social_ema_short_length = 3
    social_ema_long_length = 6
    # min_liquidity = 50000
    # trailing_stop_loss_pct = [0.94, 0.96, 0.98, 0.99]
    # trailing_stop_loss_activation_level = [1.03, 1.05, 1.07, 1.09, 1.11] #1.07

    # trailing_stop_loss_pct = 0.96
    # trailing_stop_loss_activation_level = 1.07
    
    # stop_loss_pct = [0.85, 0.875, 0.9, 0.925, 0.95]
    stop_loss_pct = 0.9

    TP1_PROFIT_PERCENTAGE = 0.07  # 50% for TP1
    TP2_PROFIT_PERCENTAGE = 0.11  # 25% for TP2
    TP3_PROFIT_PERCENTAGE = 0.25  # 25% for TP3
    
    TP1_PERCENTAGE = 0.25  # 50% for TP1
    TP2_PERCENTAGE = 0.5  # 25% for TP2
    TP3_PERCENTAGE = 0.25  # 25% for TP3

    # Constants for Stop Loss adjustments at TP levels
    TP1_STOP_LOSS_ADJUSTMENT = 1.01 # Adjust stop loss to entry price at TP1
    TP2_STOP_LOSS_ADJUSTMENT = 1.05  # Adjust stop loss to TP1 price at TP2
    TP3_STOP_LOSS_ADJUSTMENT = 1.15  # Adjust stop loss at TP3 (if needed)

    # Trigger Combinations #
    # trigger_conditions = ['social_mentions_increasing',
    #     'social_mentions_increasing&&interactions_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing&&posts_created_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing&&posts_created_increasing&&posts_active_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing&&posts_created_increasing&&posts_active_increasing&&social_dominance_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing&&posts_created_increasing&&posts_active_increasing&&social_dominance_increasing&&contributors_active_increasing',
    #     'social_mentions_increasing&&interactions_increasing&&sentiment_increasing&&posts_created_increasing&&posts_active_increasing&&social_dominance_increasing&&contributors_active_increasing&&contributors_created_increasing',
    #     'social_mentions_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over&&posts_created_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over&&posts_created_ema_cross_over&&posts_active_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over&&posts_created_ema_cross_over&&posts_active_ema_cross_over&&social_dominance_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over&&posts_created_ema_cross_over&&posts_active_ema_cross_over&&social_dominance_ema_cross_over&&contributors_active_ema_cross_over',
    #     'social_mentions_ema_cross_over&&interactions_ema_cross_over&&sentiment_ema_cross_over&&posts_created_ema_cross_over&&posts_active_ema_cross_over&&social_dominance_ema_cross_over&&contributors_active_ema_cross_over&&contributors_created_ema_cross_over'
    # ]
    trigger_conditions = encoded_conditions







    #
    # Live trading only
    #
    chain_id = ChainId.arbitrum
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(days=sma_length + 1)
    RESERVE_CURRENCY = ReserveCurrency.usdc_e


    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2023, 1, 1)
    backtest_end = datetime.datetime(2024, 5, 1)
    initial_cash = 10_000

    stop_loss_time_bucket = TimeBucket.h1


# parameters = StrategyParameters.from_class(Parameters)  # Convert to AttributedDict to easier typing with dot notation
parameters = StrategyParameters.from_class(Parameters, grid_search=True)  # Convert to AttributedDict to easier typing with dot notation


# # Trading pairs and market data
# 
# - Get a list of ERC-20 tokens we are going to trade on Polygon
# - Trading pairs are automatically mapped to the best volume /USDC or /WMATIC pair
#     - Limited to current market information - no historical volume/liquidity analyses performed here
# - This data loading method caps out at 75 trading pairs

# In[40]:


import pandas as pd

def get_highest_liquidity_pairs(pairs_df, sort_by='total_liquidity_30d'):
    """
    Get pairs with the highest liquidity based on a specified criterion.
    
    Parameters:
    - pairs_df: DataFrame containing pairs information.
    - sort_by: Criterion for sorting and selecting the highest liquidity pair.
               Accepts 'total_liquidity_30d' or 'all_time_volume'.
    
    Returns:
    - DataFrame containing pairs with the highest liquidity for each base token.
    """
    # Calculate total liquidity for the last 30 days if not already present
    if 'total_liquidity_30d' not in pairs_df.columns:
        pairs_df['total_liquidity_30d'] = pairs_df['buy_volume_30d'] + pairs_df['sell_volume_30d']
    
    # Calculate total volume of all time if requested and not already present
    if sort_by == 'all_time_volume' and 'total_volume_all_time' not in pairs_df.columns:
        pairs_df['total_volume_all_time'] = pairs_df['buy_volume_all_time'] + pairs_df['sell_volume_all_time']
    
    # Determine the sorting column based on the sort_by parameter
    if sort_by == 'all_time_volume':
        sort_column = 'total_volume_all_time'
    else:
        sort_column = 'total_liquidity_30d'
    
    # Sort the DataFrame by 'base_token_symbol' and the specified sorting column in descending order
    pairs_df_sorted = pairs_df.sort_values(by=['base_token_symbol', sort_column], ascending=[True, False])
    
    # Drop duplicates, keeping the first entry for each 'base_token_symbol'
    highest_liquidity_pairs_df = pairs_df_sorted.drop_duplicates(subset=['base_token_symbol'], keep='first')
    
    # Reset index if needed
    highest_liquidity_pairs_df.reset_index(drop=True, inplace=True)
    
    return highest_liquidity_pairs_df

# Example usage:
# Assuming pairs_df is your DataFrame and it's already loaded
# highest_liquidity_pairs_30d = get_highest_liquidity_pairs(pairs_df, sort_by='total_liquidity_30d')
# highest_liquidity_pairs_all_time = get_highest_liquidity_pairs(pairs_df, sort_by='all_time_volume')


# In[41]:


from tradingstrategy.universe import Universe
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradeexecutor.strategy.pandas_trader.alternative_market_data import resample_multi_pair
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.pair import filter_for_base_tokens, PandasPairUniverse, StablecoinFilteringMode, \
    filter_for_stablecoins
from tradingstrategy.client import Client

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradeexecutor.strategy.execution_context import ExecutionContext, notebook_execution_context
from tradeexecutor.strategy.universe_model import UniverseOptions


WETH = '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'.lower()
USDC = '0xaf88d065e77c8cC2239327C5EDb3A432268e5831'.lower()
USDCE = '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8'.lower()
USDT = '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9'.lower()
DAI = '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1'.lower()

PENDLE = "0x0c880f6761F1af8d9Aa9C466984b80DAb9a8c9e8".lower()

# We care only Quickswap and Uniswap v3 pairs.
# ApeSwap is mostly dead, but listed many bad tokens 
# in the past, so it is good to include in the sample set.
SUPPORTED_DEXES = {
    "uniswap-v3",
    "uniswap-v2",
    "sushi",
    "camelot"
}

# Get the token list of everything in the CSV + hardcoded WMATIC and other stablecoins
custom_data_token_set = {WETH, USDC, USDCE, USDT, DAI, PENDLE} | set(arb_erc20_address_list)


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe."""

    # Build a list of unique tokens
    # polygon_token_list = {TokenTuple(ChainId.polygon, address) for address in polygon_erc20_address_list}

    # strategy_universe = create_trading_universe_for_tokens(
    #     client,
    #     execution_context,
    #     universe_options,
    #     Parameters.candle_time_bucket,
    #     polygon_token_list,
    #     reserve_token="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # USDC.e bridged on Polygon
    #     intermediate_token="0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",  # WMATIC on Polygon
    #     volume_30d_threshold_today=0,
    #     stop_loss_time_bucket=parameters.stop_loss_time_bucket
    # )
    # return strategy_universe
    start_at = pd.Timestamp("2023-01-01")
    end_at = pd.Timestamp(universe_options.end_at)

    print(f"Backtesting {start_at} - {end_at}")

    chain_id = Parameters.chain_id

    exchange_universe = client.fetch_exchange_universe()
    exchange_universe = exchange_universe.limit_to_chains({Parameters.chain_id}).limit_to_slugs(SUPPORTED_DEXES)
    print(f"We support {exchange_universe.get_exchange_count()} DEXes")

    pairs_df = client.fetch_pair_universe().to_pandas()

    liquidity_df = pd.read_parquet(liquidity_output_fname)
    price_df = pd.read_parquet(price_output_fname)

    # When reading from Parquet file, we need to deal with indexing by hand
    liquidity_df.index = pd.DatetimeIndex(liquidity_df.timestamp)
    price_df.index = pd.DatetimeIndex(price_df.timestamp)

    print(f"Prefilter data contains {len(liquidity_df):,} liquidity samples dn {len(price_df):,} OHLCV candles")

    # Crop price and liquidity data to our backtesting range
    price_df = price_df.loc[(price_df.timestamp >= start_at) & (price_df.timestamp <= end_at)]
    liquidity_df = liquidity_df.loc[(liquidity_df.timestamp >= start_at) & (liquidity_df.timestamp <= end_at)]

    # Prefilter for more liquidity conditions
    liquidity_per_pair = liquidity_df.groupby(liquidity_df.pair_id)
    print(f"Chain {chain_id.name} has liquidity data for {len(liquidity_per_pair.groups)}")

    passed_pair_ids = set()
    for pair_id, pair_df in liquidity_per_pair:
        if pair_df["high"].max() > Parameters.min_liquidity:
            passed_pair_ids.add(pair_id)

    pairs_df = pairs_df.loc[pairs_df.pair_id.isin(passed_pair_ids)]
    print(f"After liquidity filter {Parameters.min_liquidity:,} USD we have {len(pairs_df)} trading pairs")

    price_per_pair = price_df.groupby(price_df.pair_id)
    passed_pair_ids = set()
    for pair_id, pair_df in price_per_pair:
        if pair_df["high"].max() < Parameters.max_price and pair_df["low"].min() > Parameters.min_price:
            passed_pair_ids.add(pair_id)

    pairs_df = pairs_df.loc[pairs_df.pair_id.isin(passed_pair_ids)]
    print(f"After broken price unit filter we have {len(pairs_df)} trading pairs")

    allowed_exchange_ids = set(exchange_universe.exchanges.keys())
    pairs_df = pairs_df.loc[pairs_df.exchange_id.isin(allowed_exchange_ids)]
    print(f"After DEX filter we have {len(pairs_df)} trading pairs")

    # Do cross-section of Polygon tokens from custom data 
    pairs_df = filter_for_base_tokens(pairs_df, custom_data_token_set)
    pairs_df = filter_for_stablecoins(pairs_df, StablecoinFilteringMode.only_volatile_pairs)
    print(f"After custom data ERC-20 token address cross section filter we have {len(pairs_df)} matching trading pairs")

    pairs_df = get_highest_liquidity_pairs(pairs_df, sort_by='all_time_volume')

    # highest_liquidity_pairs_30d = get_highest_liquidity_pairs(pairs_df, sort_by='total_liquidity_30d')
    # highest_liquidity_pairs_all_time.to_csv('../data/highest_liquidity_pairs_all_time.csv')
    # highest_liquidity_pairs_30d.to_csv('../data/highest_liquidity_pairs_30d.csv')
    # pairs_df.to_csv('../data/pairs_df_polygon.csv')

    # Resample strategy decision candles to daily
    daily_candles = resample_multi_pair(price_df, Parameters.candle_time_bucket)
    daily_candles["timestamp"] = daily_candles.index

    print(f"After downsampling we have {len(daily_candles)} OHLCV candles and {len(liquidity_df)} liquidity samples")
    candle_universe = GroupedCandleUniverse(
        daily_candles,
        time_bucket=Parameters.candle_time_bucket,
        forward_fill=True  # Forward will should make sure we can always calculate RSI, other indicators
    )

    liquidity_universe = GroupedLiquidityUniverse(liquidity_df)

    # The final trading pair universe contains metadata only for pairs that passed
    # our filters
    pairs_universe = PandasPairUniverse(pairs_df, exchange_universe=exchange_universe)
    stop_loss_candle_universe = GroupedCandleUniverse(price_df)

    data_universe = Universe(
        time_bucket=Parameters.candle_time_bucket,
        liquidity_time_bucket=Parameters.candle_time_bucket,
        exchange_universe=exchange_universe,
        pairs=pairs_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
        chains={Parameters.chain_id},
        forward_filled=True,
    )
    # print(f"Get usdc token {pairs_universe.get_token(USDC)}")
    reserve_asset = translate_token(pairs_universe.get_token(USDCE))

    _strategy_universe = TradingStrategyUniverse(
        data_universe=data_universe,
        backtest_stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
        backtest_stop_loss_candles=stop_loss_candle_universe,
        reserve_assets={reserve_asset},
        required_history_period=parameters.required_history_period
    )

    return _strategy_universe


strategy_universe = create_trading_universe(
    None,
    client,
    notebook_execution_context,
    UniverseOptions.from_strategy_parameters_class(Parameters, notebook_execution_context)
)

broken_trading_pairs = set()

#
# Extra sanity checks
# 
# Ru some extra sanity check for small cap tokens
#

print("Checking trading pair quality")
print("-" * 80)

for pair in strategy_universe.iterate_pairs():
    reason = strategy_universe.get_trading_broken_reason(pair, min_candles_required=41, min_price=0.00000000001)
    if reason:
        print(f"FAIL: {pair} with base token {pair.base.address} may be problematic: {reason}")
        broken_trading_pairs.add(pair)
    else:
        print(f"OK: {pair} included in the backtest")

print(f"Total {len(broken_trading_pairs)} broken trading pairs detected, having {strategy_universe.get_pair_count() - len(broken_trading_pairs)} good pairs left to trade")


# # Indicators
# 
# - We use `pandas_ta` Python package to calculate technical indicators
# - These indicators are precalculated and cached on the disk
# - This includes caching our custom made indicators, so we only calculate them once

# In[42]:


import pandas as pd
import pandas_ta
import datetime
from tradeexecutor.analysis.regime import Regime
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.crossover import contains_cross_over, contains_cross_under


def rsi_safe(close_series: pd.Series, length: int):
    """RSI made length safe: Work around tokens that appear only for few hours.
    
    Used to determine a trading pair momentum.
    """
    if len(close_series) > length:
        return pandas_ta.rsi(close_series, length)
    else:
        # The token did not trade long enough we could have ever calculated RSI
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))

def daily_adx(open, high, low, close, length):
    adx_df = pandas_ta.adx(
        close=close,
        high=high,
        low=low,
        length=length
    )
    return adx_df

def regime(open, high, low, close, length, regime_threshold) -> pd.Series:
    """A regime filter based on ADX indicator.

    Get the trend of BTC applying ADX on a daily frame.
    
    - -1 is bear
    - 0 is sideways
    - +1 is bull
    """
    adx_df = daily_adx(open, high, low, close, length)
    def regime_filter(row):
        # ADX, DMP, # DMN
        average_direction_index, directional_momentum_positive, directional_momentum_negative = row.values
        if directional_momentum_positive > regime_threshold:
            return Regime.bull.value
        elif directional_momentum_negative > regime_threshold:
            return Regime.bear.value
        else:
            return Regime.crab.value
    regime_signal = adx_df.apply(regime_filter, axis="columns")    
    return regime_signal

def sma_guard(close_series: pd.Series, length: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA) on close prices with a check for series length.
    
    Args:
        close_series (pd.Series): Series of close prices.
        length (int): The number of periods to calculate the SMA over.
        
    Returns:
        pd.Series: Series containing the SMA of the provided close prices.
    """
    if len(close_series) >= length:
        return close_series.rolling(window=length).mean()
    else:
        # Return an empty Series if there aren't enough points to calculate SMA
        return pd.Series(dtype="float64", index=close_series.index)

def add_social_mentions(pair: TradingPairIdentifier) -> pd.Series:
    """Add social_mentions as social_mentions to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["social_mentions"]
    # Clean up duplicate dates in datetime index
    social_mentions = series[~series.index.duplicated()].rename("social_mentions")
    
    return social_mentions

def calculate_social_mentions_bbands(pair: TradingPairIdentifier, length: int, std: int) -> pd.DataFrame:
    """Calculate Bollinger Bands for social_mentions."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["social_mentions"]
    # Clean up duplicate dates in datetime index
    series_without_duplicates = series[~series.index.duplicated()]
    
    # Calculate Bollinger Bands
    bb = pandas_ta.bbands(series_without_duplicates, length=40, std=2)

    bb = bb.rename(columns={
        'BBL_40_2.0': 'BBL_mentions_40_2.0',
        'BBM_40_2.0': 'BBM_mentions_40_2.0',
        'BBU_40_2.0': 'BBU_mentions_40_2.0',
        'BBB_40_2.0': 'BBB_mentions_40_2.0',
        'BBP_40_2.0': 'BBP_mentions_40_2.0'
    })
    
    return bb

def add_social_interactions(pair: TradingPairIdentifier) -> pd.Series:
    """Add coin_id_count as social_mentions to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["interactions"]
    # Clean up duplicate dates in datetime index
    social_mentions = series[~series.index.duplicated()].rename("interactions")
    
    return social_mentions

def calculate_social_interactions_bbands(pair: TradingPairIdentifier, length: int, std: int) -> pd.DataFrame:
    """Calculate Bollinger Bands for social_mentions."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["interactions"]
    # Clean up duplicate dates in datetime index
    series_without_duplicates = series[~series.index.duplicated()]
    
    # Calculate Bollinger Bands
    bb = pandas_ta.bbands(series_without_duplicates, length=40, std=2)

    bb = bb.rename(columns={
        'BBL_40_2.0': 'BBL_interactions_40_2.0',
        'BBM_40_2.0': 'BBM_interactions_40_2.0',
        'BBU_40_2.0': 'BBU_interactions_40_2.0',
        'BBB_40_2.0': 'BBB_interactions_40_2.0',
        'BBP_40_2.0': 'BBP_interactions_40_2.0'
    })
    
    return bb

def add_social_sentiment(pair: TradingPairIdentifier) -> pd.Series:
    """Add coin_id_count as social_mentions to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["sentiment"]
    # Clean up duplicate dates in datetime index
    social_mentions = series[~series.index.duplicated()].rename("sentiment")
    
    return social_mentions

def calculate_social_sentiment_bbands(pair: TradingPairIdentifier, length: int, std: int) -> pd.DataFrame:
    """Calculate Bollinger Bands for social_mentions."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair_sentiment_1w = custom_data_group.get_group(contract_address)
    series = per_pair_sentiment_1w["sentiment"]
    # Clean up duplicate dates in datetime index
    series_without_duplicates = series[~series.index.duplicated()]
    
    # Calculate Bollinger Bands
    bb = pandas_ta.bbands(series_without_duplicates, length=length, std=std)
    bb = bb.rename(columns={
        'BBL_40_2.0': 'BBL_sentiment_40_2.0',
        'BBM_40_2.0': 'BBM_sentiment_40_2.0',
        'BBU_40_2.0': 'BBU_sentiment_40_2.0',
        'BBB_40_2.0': 'BBB_sentiment_40_2.0',
        'BBP_40_2.0': 'BBP_sentiment_40_2.0'
    })
    
    return bb

def add_alt_rank(pair: TradingPairIdentifier) -> pd.Series:
    """Add alt_rank to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["alt_rank"]
    # Clean up duplicate dates in datetime index
    alt_rank = series[~series.index.duplicated()].rename("alt_rank")
    return alt_rank

def add_posts_created(pair: TradingPairIdentifier) -> pd.Series:
    """Add posts_created to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["posts_created"]
    # Clean up duplicate dates in datetime index
    posts_created = series[~series.index.duplicated()].rename("posts_created")
    return posts_created

def add_posts_active(pair: TradingPairIdentifier) -> pd.Series:
    """Add posts_active to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["posts_active"]
    # Clean up duplicate dates in datetime index
    posts_active = series[~series.index.duplicated()].rename("posts_active")
    return posts_active

def add_social_dominance(pair: TradingPairIdentifier) -> pd.Series:
    """Add social_dominance to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["social_dominance"]
    # Clean up duplicate dates in datetime index
    social_dominance = series[~series.index.duplicated()].rename("social_dominance")
    return social_dominance

def add_contributors_active(pair: TradingPairIdentifier) -> pd.Series:
    """Add contributors_active to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["contributors_active"]
    # Clean up duplicate dates in datetime index
    contributors_active = series[~series.index.duplicated()].rename("contributors_active")
    return contributors_active

def add_contributors_created(pair: TradingPairIdentifier) -> pd.Series:
    """Add contributors_created to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["contributors_created"]
    # Clean up duplicate dates in datetime index
    contributors_created = series[~series.index.duplicated()].rename("contributors_created")
    return contributors_created

def add_volatility(pair: TradingPairIdentifier) -> pd.Series:
    """Add volatility to the dataset."""
    contract_address = pair.base.address  # ERC-20 address of the base token of the trading pair
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["volatility"]
    # Clean up duplicate dates in datetime index
    volatility = series[~series.index.duplicated()].rename("volatility")
    return volatility


def sma_volume(open, high, low, close, volume) -> pd.DataFrame:
    """Volume SMA"""
    original_df = pd.DataFrame({
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }) 
    rolling_vol = original_df['volume'].rolling(window=20).mean()  
    return rolling_vol

def add_posts_active_with_emas(pair: TradingPairIdentifier) -> pd.DataFrame:
    """Add posts_active and its EMAs to the dataset."""
    contract_address = pair.base.address
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair["posts_active"]
    series_clean = series[~series.index.duplicated()].rename("posts_active")
    
    # Calculate EMAs
    ema_3d = series_clean.ewm(span=3, adjust=False).mean().rename("posts_active_ema_3d")
    ema_6d = series_clean.ewm(span=6, adjust=False).mean().rename("posts_active_ema_6d")
    
    # Combine into a DataFrame
    combined_df = pd.concat([series_clean, ema_3d, ema_6d], axis=1)

    print('combined_df cols', combined_df.columns)
    
    return combined_df

def add_metric(pair: TradingPairIdentifier, metric_name: str) -> pd.Series:
    """Add a specific metric to the dataset."""
    contract_address = pair.base.address
    per_pair = custom_data_group.get_group(contract_address)
    series = per_pair[metric_name]
    # Clean up duplicate dates in datetime index
    metric_series = series[~series.index.duplicated()].rename(metric_name)
    return metric_series

# Function to calculate the short and long EMAs for each metric
def calculate_metric_emas(pair: TradingPairIdentifier, metric_name: str, short_length: int, long_length: int) -> pd.DataFrame:
    """Calculate short and long EMAs for a specific metric."""
    metric_series = add_metric(pair, metric_name)
    ema_short = metric_series.ewm(span=short_length, adjust=False).mean().rename(f"{metric_name}_ema_short")
    ema_long = metric_series.ewm(span=long_length, adjust=False).mean().rename(f"{metric_name}_ema_long")
    emas_df = pd.concat([ema_short, ema_long], axis=1)
    return emas_df

# def calculate_metric_emas_with_crossover(pair: TradingPairIdentifier, metric_name: str, short_length: int, long_length: int) -> pd.DataFrame:
#     """Calculate short and long EMAs for a specific metric and determine if a crossover occurred."""
#     # Get the DataFrame with short and long EMAs
#     ema_df = calculate_metric_emas(pair, metric_name, short_length, long_length)
    
#     # Extract the short and long EMA series
#     ema_short = ema_df[f"{metric_name}_ema_short"]
#     ema_long = ema_df[f"{metric_name}_ema_long"]
    
#     # Use the predefined function to check for crossover
#     try:
#         crossover, crossover_index = contains_cross_over(
#             ema_short,
#             ema_long,
#             lookback_period=2,
#             must_return_index=True
#         )
#         print('crossover series', crossover)
#     except Exception as e:
#         # Handle exceptions if contains_cross_over fails
#         print(f"Error calculating crossover for {metric_name}: {e}")
#         crossover = pd.Series(False, index=ema_short.index)
#         crossover_index = -1  # Default value indicating no crossover

#     # Rename the crossover series and add it to the DataFrame
#     crossover = crossover.rename(f"{metric_name}_crossover")
#     emas_with_crossover_df = pd.concat([ema_df, crossover], axis=1)
    
#     # Optionally, you can add the crossover index to the DataFrame if needed
#     # emas_with_crossover_df[f"{metric_name}_crossover_index"] = crossover_index

#     return emas_with_crossover_df


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()
    # IndicatorSource.

    indicators.add(
        "rsi",
        rsi_safe,
        {"length": parameters.rsi_length},
        IndicatorSource.close_price,
    )    

    
    indicators.add(
        "sma_volume",
        sma_volume,
        {},
        IndicatorSource.ohlcv,
    )
    
    indicators.add(
        "sma",
        sma_guard,
        {"length": parameters.sma_length},  
        IndicatorSource.close_price,
    )  

    # indicators.add(
    #     "social_mentions",
    #     add_social_mentions,
    #     {},  # No parameters needed for this custom indicator
    #     IndicatorSource.external_per_pair,
    # )
    
    # Add Bollinger Bands for social_mentions as another indicator
    # indicators.add(
    #     "social_mentions_bbands",
    #     calculate_social_mentions_bbands,
    #     {"length": parameters.bb_length, "std": parameters.bb_std},  # No parameters for our custom indicator
    #     IndicatorSource.external_per_pair,
    # )

    # indicators.add(
    #     "interactions",
    #     add_social_interactions,
    #     {},  # No parameters needed for this custom indicator
    #     IndicatorSource.external_per_pair,
    # )
    
    # Add Bollinger Bands for social_mentions as another indicator
    # indicators.add(
    #     "interactions_bbands",
    #     calculate_interactions_bbands,
    #     {"length": parameters.bb_length, "std": parameters.bb_std},  # No parameters for our custom indicator
    #     IndicatorSource.external_per_pair,
    # )

    # indicators.add(
    #     "sentiment",
    #     add_social_sentiment,
    #     {},  # No parameters needed for this custom indicator
    #     IndicatorSource.external_per_pair,
    # )

    # indicators.add(
    #     "sentiment_bb_bands",
    #     calculate_social_sentiment_bbands,
    #     {"length": parameters.bb_length, "std": parameters.bb_std},
    #     IndicatorSource.external_per_pair,
    # )

    #Social Metrics
    # Add original value indicators
    social_metrics = [
        "posts_created",
        "social_dominance",
        "contributors_active",
        "contributors_created",
        "volatility",
        "posts_active",
        "sentiment",
        "interactions",
        "social_mentions"
    ]

    for metric in social_metrics:
        indicators.add(
            metric,
            add_metric,
            {"metric_name": metric},  # Pass the metric name as a parameter
            IndicatorSource.external_per_pair,
        )

        # Add EMA indicators
        indicators.add(
            f"{metric}_emas",
            calculate_metric_emas,
            {
                "metric_name": metric,
                "short_length": parameters.social_ema_short_length,
                "long_length": parameters.social_ema_long_length
            },
            IndicatorSource.external_per_pair,
        )

    # Example usage in create_indicators function
    # indicators.add(
    #     "posts_active_with_emas",
    #     add_posts_active_with_emas,
    #     {},  # No parameters needed for this custom indicator
    #     IndicatorSource.external_per_pair,
    # )

    # indicators.add("alt_rank", add_alt_rank, {},
    #            IndicatorSource.external_per_pair)

    # indicators.add("posts_created", add_posts_created, {},
    #            IndicatorSource.external_per_pair)

    # # indicators.add("posts_active", add_posts_active, {},
    # #            IndicatorSource.external_per_pair)

    # indicators.add("social_dominance", add_social_dominance, {},
    #            IndicatorSource.external_per_pair)

    # indicators.add("contributors_active", add_contributors_active, {},
    #            IndicatorSource.external_per_pair)

    # indicators.add("contributors_created", add_contributors_created, {},
    #            IndicatorSource.external_per_pair)

    # indicators.add("volatility", add_volatility, {},
    #            IndicatorSource.external_per_pair)

    return indicators


# # Trading algorithm
# 
# - Describe out trading strategy as code

# In[49]:


from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, StrategyInputIndicators
from tradeexecutor.strategy.weighting import weight_by_1_slash_n
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.pandas_trader.strategy_input import IndicatorDataNotFoundWithinDataTolerance
from tradeexecutor.utils.crossover import contains_cross_over, contains_cross_under
from pandas_ta.overlap import ema


def decode_conditions(encoded_string):
                # Splits the string back into the list of conditions using '&' as the delimiter
    return encoded_string.split('&')

def update_tp_level(tp_levels_reached, position_id, tp_level, reached=True):
    """
    Update the TP level reached status for a given position in the provided dictionary.
    
    Parameters:
    - tp_levels_reached: Dictionary tracking TP levels for positions.
    - position_id: Unique identifier for the position.
    - tp_level: The TP level being updated (e.g., 'tp1', 'tp2').
    - reached: Boolean indicating whether the TP level has been reached.
    """
    if position_id not in tp_levels_reached:
        tp_levels_reached[position_id] = {}
    tp_levels_reached[position_id][tp_level] = reached

def is_tp_level_reached(tp_levels_reached, position_id, tp_level):
    """
    Check if a TP level has been reached for a given position in the provided dictionary.
    
    Parameters:
    - tp_levels_reached: Dictionary tracking TP levels for positions.
    - position_id: Unique identifier for the position.
    - tp_level: The TP level to check (e.g., 'tp1', 'tp2').
    
    Returns:
    - Boolean indicating whether the TP level has been reached.
    """
    return tp_levels_reached.get(position_id, {}).get(tp_level, False)


def is_volume_increasing(current_volume, volume_sma):
    """Check if the current volume is significantly higher than the SMA."""
    return current_volume > volume_sma


tp_levels_reached = {}

# def get_social_data_analysis(pair: TradingPairIdentifier, indicators: StrategyInputIndicators) -> dict:
#     social_data_analysis = {}

#     social_mentions = indicators.get_indicator_value("social_mentions", pair=pair)
#     social_mentions_previous_value = indicators.get_indicator_value("social_mentions", pair=pair, index=-2)
#     social_data_analysis["social_mentions"] = social_mentions
#     social_data_analysis["social_mentions_previous"] = social_mentions_previous_value
#     social_data_analysis["social_mentions_increasing"] = social_mentions is not None and social_mentions_previous_value is not None and social_mentions > social_mentions_previous_value

#     interactions = indicators.get_indicator_value("interactions", pair=pair)
#     interactions_previous_value = indicators.get_indicator_value("interactions", pair=pair, index=-2)
#     social_data_analysis["interactions"] = interactions
#     social_data_analysis["interactions_previous"] = interactions_previous_value
#     social_data_analysis["interactions_increasing"] = interactions is not None and social_interactions_previous_value is not None and social_interactions > social_interactions_previous_value

#     sentiment = indicators.get_indicator_value("sentiment", pair=pair)
#     sentiment_previous_value = indicators.get_indicator_value("sentiment", pair=pair, index=-2)
#     social_data_analysis["sentiment"] = sentiment
#     social_data_analysis["sentiment_previous"] = sentiment_previous_value
#     social_data_analysis["sentiment_increasing"] = sentiment is not None and sentiment_previous_value is not None and sentiment > sentiment_previous_value

#     alt_rank = indicators.get_indicator_value("alt_rank", pair=pair)
#     alt_rank_previous_value = indicators.get_indicator_value("alt_rank", pair=pair, index=-2)
#     social_data_analysis["alt_rank"] = alt_rank
#     social_data_analysis["alt_rank_previous"] = alt_rank_previous_value
#     social_data_analysis["alt_rank_increasing"] = alt_rank is not None and alt_rank_previous_value is not None and alt_rank > alt_rank_previous_value

#     posts_created = indicators.get_indicator_value("posts_created", pair=pair)
#     posts_created_previous_value = indicators.get_indicator_value("posts_created", pair=pair, index=-2)
#     social_data_analysis["posts_created"] = posts_created
#     social_data_analysis["posts_created_previous"] = posts_created_previous_value
#     social_data_analysis["posts_created_increasing"] = posts_created is not None and posts_created_previous_value is not None and posts_created > posts_created_previous_value

#     posts_active = indicators.get_indicator_value("posts_active", pair=pair)
#     posts_active_previous_value = indicators.get_indicator_value("posts_active", pair=pair, index=-2)
#     social_data_analysis["posts_active"] = posts_active
#     social_data_analysis["posts_active_previous"] = posts_active_previous_value
#     social_data_analysis["posts_active_increasing"] = posts_active is not None and posts_active_previous_value is not None and posts_active > posts_active_previous_value

    # social_dominance = indicators.get_indicator_value("social_dominance", pair=pair)
    # social_dominance_previous_value = indicators.get_indicator_value("social_dominance", pair=pair, index=-2)
    # social_data_analysis["social_dominance"] = social_dominance
    # social_data_analysis["social_dominance_previous"] = social_dominance_previous_value
    # social_data_analysis["social_dominance_increasing"] = social_dominance is not None and social_dominance_previous_value is not None and social_dominance > social_dominance_previous_value

    # contributors_active = indicators.get_indicator_value("contributors_active", pair=pair)
    # contributors_active_previous_value = indicators.get_indicator_value("contributors_active", pair=pair, index=-2)
    # social_data_analysis["contributors_active"] = contributors_active
    # social_data_analysis["contributors_active_previous"] = contributors_active_previous_value
    # social_data_analysis["contributors_active_increasing"] = contributors_active is not None and contributors_active_previous_value is not None and contributors_active > contributors_active_previous_value

    # contributors_created = indicators.get_indicator_value("contributors_created", pair=pair)
    # contributors_created_previous_value = indicators.get_indicator_value("contributors_created", pair=pair, index=-2)
    # social_data_analysis["contributors_created"] = contributors_created
    # social_data_analysis["contributors_created_previous"] = contributors_created_previous_value
    # social_data_analysis["contributors_created_increasing"] = contributors_created is not None and contributors_created_previous_value is not None and contributors_created > contributors_created_previous_value

    # volatility = indicators.get_indicator_value("volatility", pair=pair)
    # volatility_previous_value = indicators.get_indicator_value("volatility", pair=pair, index=-2)
    # social_data_analysis["volatility"] = volatility
    # social_data_analysis["volatility_previous"] = volatility_previous_value
    # social_data_analysis["volatility_increasing"] = volatility is not None and volatility_previous_value is not None and volatility > volatility_previous_value

    # increasing_factors = []
    # non_null_factors = 0
    # for key in ["social_mentions_increasing", "social_interactions_increasing", "sentiment_increasing", "alt_rank_increasing", "posts_created_increasing", "posts_active_increasing", "social_dominance_increasing", "contributors_active_increasing", "contributors_created_increasing", "volatility_increasing"]:
    # for key in ["social_mentions_increasing", "social_interactions_increasing", "sentiment_increasing",  "posts_created_increasing", "posts_active_increasing", "social_dominance_increasing", "contributors_active_increasing", "contributors_created_increasing"]:
    #     current_value = social_data_analysis.get(key.replace("_increasing", ""), None)
    #     previous_value = social_data_analysis.get(key.replace("_increasing", "_previous"), None)

    #     if current_value is not None and previous_value is not None:
    #         non_null_factors += 1
    #         increasing_factors.append(social_data_analysis.get(key, False))

    # num_increasing_factors = sum(increasing_factors)
    # non_null_factors = len(increasing_factors)
    # print('increasing_factors', num_increasing_factors)
    # print('non_null_factors', non_null_factors)

    # print(f"social_data_analysis", social_data_analysis)
    # if non_null_factors > 0:
    #     print('num_increasing_factors:', num_increasing_factors)
    #     print('non_null_factors:', non_null_factors)
    #     print("ratio_increasing", num_increasing_factors / non_null_factors)
    #     social_data_analysis["ratio_increasing"] = num_increasing_factors / non_null_factors
    #     social_data_analysis["ratio_increasing"] = None

    # return social_data_analysis


# def get_social_data_analysis(pair: TradingPairIdentifier, indicators: StrategyInputIndicators) -> dict:
#     social_data_analysis = {}

#     for metric in ["social_mentions", "social_interactions", "sentiment", "posts_created", "posts_active", "social_dominance", "contributors_active", "contributors_created"]:
#         metric_series = indicators.get_indicator_series(metric, pair=pair)
#         if metric_series is not None and not metric_series.empty:
#             ema_3d = ema(metric_series, length=3)  # Calculate 3-day EMA
#             ema_6d = ema(metric_series, length=6)  # Calculate 3-day EMA
#             latest_value = metric_series.iloc[-1]
#             social_data_analysis[metric] = latest_value
#             social_data_analysis[f"{metric}_ema_3d"] = ema_3d
#             social_data_analysis[f"{metric}_ema_6d"] = ema_6d
            

#             social_data_analysis[f"{metric}_increasing"] = latest_value > ema_6d if latest_value is not None and ema_6d is not None else False
#             print(f"ema3d type {type(ema_3d)}")
#             print(f"ema6d type {type(ema_6d)}")
#             try: 
#                 crossover_mentions, mentions_cross_over_index = contains_cross_over(
#                     ema_3d ,
#                     ema_6d, 
#                     lookback_period=2,
#                     must_return_index=True
#                 )
#             except:
#                 crossover_mentions = None
#                 mentions_cross_over_index = None
#             social_data_analysis[f"{metric}_ema_cross_over"] = True if (crossover_mentions and mentions_cross_over_index == -1) else False
#         else:
#             social_data_analysis[metric] = None
#             social_data_analysis[f"{metric}_ema_3d"] = None
#             social_data_analysis[f"{metric}_ema_6d"] = None
#             social_data_analysis[f"{metric}_increasing"] = False
#             social_data_analysis[f"{metric}_ema_cross_over"] = False

#     return social_data_analysis

def get_social_data_analysis(pair: TradingPairIdentifier, indicators: StrategyInputIndicators) -> dict:
    social_data_analysis = {}

    for metric in ["social_mentions", "interactions", "sentiment", "posts_created", "posts_active", "social_dominance", "contributors_active", "contributors_created"]:
        # Get the latest value of the metric
        latest_value = indicators.get_indicator_value(metric, pair=pair)
        social_data_analysis[metric] = latest_value

        # Get the 3-day and 6-day EMAs from the indicators
        ema_3d = indicators.get_indicator_series(f"{metric}_emas", column=f"{metric}_ema_short", pair=pair)
        ema_6d = indicators.get_indicator_series(f"{metric}_emas", column=f"{metric}_ema_long", pair=pair)
        social_data_analysis[f"{metric}_ema_short"] = ema_3d
        social_data_analysis[f"{metric}_ema_long"] = ema_6d

        # Determine if the latest value is increasing compared to the 6-day EMA
        social_data_analysis[f"{metric}_increasing"] = latest_value > ema_6d.iloc[-1] if latest_value is not None and ema_6d is not None else False

        # Check for crossover using the predefined function
        try:
            crossover, crossover_index = contains_cross_over(
                ema_3d,
                ema_6d,
                lookback_period=2,
                must_return_index=True
            )
            # Assuming crossover is a boolean Series, we want the last value
            crossover_occurred = crossover
        except Exception as e:
            crossover_occurred = False

        social_data_analysis[f"{metric}_ema_cross_over"] = crossover_occurred

    return social_data_analysis

def check_conditions(conditions, indicators):
    # Check if all conditions in the list are True in the social_data_indicators
    return all(indicators.get(condition, False) for condition in conditions)


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    

    # 
    # Decision cycle setup.
    # Read all variables we are going to use for the decisions.
    #
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    cash = position_manager.get_current_cash()

    # Another low cap problem checker.
    # Doing a bad non-liquidity trade may break the valuation calculations,
    # if a bad trading pair has slipped through our filters earlier.
    total_equity = state.portfolio.get_total_equity()
    if total_equity > 1_000_000:
        position_valuations = "\n".join([f"{p} (token {p.pair.base.address}): {p.get_value()}" for p in state.portfolio.open_positions.values()])
        raise RuntimeError(f"Portfolio total equity exceeded 1,000,000 USD. Some broken math likely happened. Total equity is {total_equity} USD.\nOpen positions:\n{position_valuations}")
        
    #
    # Trading logic
    #
    # We do some extra checks here as we are trading low quality
    # low cap tokens which often have outright malicious data for trading.
    #

    trades = []

    # print(f"Current open positions", len(state.portfolio.open_positions.values()))


    # Enable trailing stop loss after we reach the profit taking level
    #
    for position in state.portfolio.open_positions.values():

        # print(f"Current positions: {position_manager.get_current_position_for_pair(position.pair)}")
        position_id = position.position_id
        close_price = indicators.get_price(position.pair)
        #alternative version of getting profit
        position_profit_pct = position.get_unrealised_and_realised_profit_percent()

        # print(f"Comparing current price {position.get_current_price()} and the opening prie {position.get_opening_price()}")
        # print(f"current price over the opening price {position.get_current_price() / position.get_opening_price()}")
        # print(f"position_profit_pct", position_profit_pct)


        # if position.trailing_stop_loss_pct is None:
        #     close_price = indicators.get_price(pair=position.pair)
        #     print(f"close_price: {close_price}, opening_price: {position.get_opening_price()}, trailing_stop_loss_activation_level: {parameters.trailing_stop_loss_activation_level}")
        #     if close_price and close_price >= position.get_opening_price() * parameters.trailing_stop_loss_activation_level:
        #         position.trailing_stop_loss_pct = parameters.trailing_stop_loss_pct 
        # elif position.stop_loss_pct:
        #     close_price = indicators.get_price(position.pair)
        #     if close_price < position.get_opening_price() * parameters.stop_loss_pct:
        #         position_manager.close_position(position)


        initial_position_quantity = float(position.get_buy_quantity())
        position_usd_valuation = position.get_current_price() * float(position.get_available_trading_quantity())
    
        # Example usage of constants in TP1 logic
        if position_profit_pct >= parameters.TP1_PROFIT_PERCENTAGE and not is_tp_level_reached(tp_levels_reached, position_id, 'tp1'):
            # Logic for TP1
            new_stop_loss =  (position.get_opening_price() * parameters.TP1_STOP_LOSS_ADJUSTMENT) / position.get_current_price()
            trades += position_manager.adjust_position(
                position.pair, 
                dollar_delta=-position_usd_valuation * parameters.TP1_PERCENTAGE, 
                quantity_delta=-initial_position_quantity * parameters.TP1_PERCENTAGE, 
                weight=0, 
                trailing_stop_loss=0.96,
                stop_loss=new_stop_loss, 
                override_stop_loss=True
            )
            update_tp_level(tp_levels_reached, position_id, 'tp1', reached=True)

        # Similarly, update the TP2 and TP3 logic to use the constants
         # Similarly, update the TP2 and TP3 logic to use the constants
        elif position_profit_pct >= parameters.TP2_PROFIT_PERCENTAGE and not is_tp_level_reached(tp_levels_reached, position_id, 'tp2'):
            # print(f"position_profit_pct {position_profit_pct} and the parameter is {parameters.TP2_PROFIT_PERCENTAGE}")
            # print(f"Comparing current price {position.get_current_price()} and the opening prie {position.get_opening_price()} and the actual profit is {position.get_current_price() / position.get_opening_price()}")
            # Logic for TP2
            new_stop_loss =  (position.get_opening_price() * parameters.TP2_STOP_LOSS_ADJUSTMENT) / position.get_current_price()
            # print(f"New stop loss is set at {new_stop_loss} using {parameters.TP2_STOP_LOSS_ADJUSTMENT} from initial price, resulting in: {(position.get_opening_price() * parameters.TP2_STOP_LOSS_ADJUSTMENT)}")
            trades += position_manager.adjust_position(
                position.pair, 
                dollar_delta=-position_usd_valuation * parameters.TP2_PERCENTAGE, 
                quantity_delta=-initial_position_quantity * parameters.TP2_PERCENTAGE, 
                weight=0, 
                # trailing_stop_loss=0.98,
                stop_loss=new_stop_loss, 
                override_stop_loss=True
            )
            update_tp_level(tp_levels_reached, position_id, 'tp2', reached=True)

        elif position_profit_pct >= parameters.TP3_PROFIT_PERCENTAGE and not is_tp_level_reached(tp_levels_reached, position_id, 'tp3'):
            # Logic for TP3
            new_stop_loss =  (position.get_opening_price() * parameters.TP3_STOP_LOSS_ADJUSTMENT) / position.get_current_price()
            trades += position_manager.close_position(position)
            update_tp_level(tp_levels_reached, position_id, 'tp3', reached=True)

        position_manager.log(f"Current positions: {position_manager.get_current_portfolio()}")


    for pair in strategy_universe.iterate_pairs():

        if pair in broken_trading_pairs:
            # Don't even bother to try trade this
            continue

        position_for_pair = state.portfolio.get_open_position_for_pair(pair)
        # if position_for_pair is not None:
            # print(f"Price updates", position_for_pair.balance_updates)

        # Extract Social indicators here
        bollinger_bands_ma_length = parameters.bb_length
        std_dev = parameters.bb_std
        # bb_upper_mentions_column = f"BBU_mentions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_mid_mentions_column = f"BBM_mentions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_lower_mentions_column = f"BBL_mentions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming

        # bb_upper_sentiment_column = f"BBU_sentiment_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_mid_sentiment_column = f"BBM_sentiment_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_lower_sentiment_column = f"BBL_sentiment_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming

        # bb_upper_interactions_column = f"BBU_interactions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_mid_interactions_column = f"BBM_interactions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
        # bb_lower_interactions_column = f"BBL_interactions_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming

        # bb_upper_mentions_social = indicators.get_indicator_value("social_mentions_bbands", column=bb_upper_mentions_column, pair=pair)
        # bb_mid_mentions_social = indicators.get_indicator_value("social_mentions_bbands", column=bb_mid_mentions_column, pair=pair)
        # bb_lower_mentions_social = indicators.get_indicator_value("social_mentions_bbands", column=bb_lower_mentions_column, pair=pair)

        # bb_upper_sentiment_social = indicators.get_indicator_value("sentiment_bb_bands", column=bb_upper_sentiment_column, pair=pair)
        # bb_mid_sentiment_social = indicators.get_indicator_value("sentiment_bb_bands", column=bb_mid_sentiment_column, pair=pair)
        # bb_lower_sentiment_social = indicators.get_indicator_value("sentiment_bb_bands", column=bb_lower_sentiment_column, pair=pair)

        # bb_upper_interactions_social = indicators.get_indicator_value("social_interactions_bbands", column=bb_upper_interactions_column, pair=pair)
        # bb_mid_interactions_social = indicators.get_indicator_value("social_interactions_bbands", column=bb_mid_interactions_column, pair=pair)
        # bb_lower_interactions_social = indicators.get_indicator_value("social_interactions_bbands", column=bb_lower_interactions_column, pair=pair)

        # social_mentions_previous_value = indicators.get_indicator_value("social_mentions", pair=pair, index=-2)
        #Want to grab the social_mentions value and the previous value. Do this with all of the below and have the current value, the previous value and a variable indicating if they are increasing or not
        # Finally I want a number as a ratio of how many of these factors are going up at the same time. If there is 10 metrics and 7 of them are going up, then the number is 0.7
        
        # social_mentions = indicators.get_indicator_value("social_mentions", pair=pair)
        # social_interactions = indicators.get_indicator_value("interactions", pair=pair)
        # sentiment = indicators.get_indicator_value("sentiment", pair=pair)
        # bb_upper_social_series = indicators.get_indicator_series("social_mentions_bbands", column=bb_upper_mentions_column, pair=pair)
        # social_mentions_series = indicators.get_indicator_series("social_mentions", pair=pair)

        # posts_active_with_emas3d = indicators.get_indicator_series('posts_active_with_emas', column='posts_active_ema_3d', pair=pair)
        # posts_active_with_ema6d = indicators.get_indicator_series('posts_active_emas', column='posts_active_ema_6d', pair=pair)
        # print('posts_active_with_emas3d', posts_active_with_emas3d)
        # print('posts_active_with_ema6d', posts_active_with_ema6d)

        # try:
        #     alt_rank_series = indicators.get_indicator_series("alt_rank", pair=pair)
        #     posts_created_series = indicators.get_indicator_series("posts_created", pair=pair)
        #     posts_active_series = indicators.get_indicator_series("posts_active", pair=pair)
        #     social_dominance_series = indicators.get_indicator_series("social_dominance", pair=pair)
        #     contributors_active_series = indicators.get_indicator_series("contributors_active", pair=pair)
        #     contributors_created_series = indicators.get_indicator_series("contributors_created", pair=pair)
        #     volatility_series = indicators.get_indicator_series("volatility", pair=pair)
        #     sentiment_series = indicators.get_indicator_series("volatility", pair=pair)

        #     slow_ema_sentiment = ema(sentiment_series, length=6)
        #     fast_ema_sentiment = ema(sentiment_series, length=3)
        #     slow_ema_latest_sentiment = slow_ema_sentiment.iloc[-1]
        #     fast_ema_latest_sentiment = fast_ema_sentiment.iloc[-1]
            
        #     slow_ema_alt_rank = ema(alt_rank_series, length=6)
        #     fast_ema_alt_rank = ema(alt_rank_series, length=3)
        #     slow_ema_posts_created = ema(posts_created_series, length=6)
        #     fast_ema_posts_created = ema(posts_created_series, length=3)
        #     slow_ema_posts_active = ema(posts_active_series, length=6)
        #     fast_ema_posts_active = ema(posts_active_series, length=3)
        #     slow_ema_social_dominance = ema(social_dominance_series, length=6)
        #     fast_ema_social_dominance = ema(social_dominance_series, length=3)
        #     slow_ema_contributors_active = ema(contributors_active_series, length=6)
        #     fast_ema_contributors_active = ema(contributors_active_series, length=3)
        #     slow_ema_contributors_created = ema(contributors_created_series, length=6)
        #     fast_ema_contributors_created = ema(contributors_created_series, length=3)
        #     slow_ema_volatility = ema(volatility_series, length=6)
        #     fast_ema_volatility = ema(volatility_series, length=3)

        #     slow_ema_latest_alt_rank = slow_ema_alt_rank.iloc[-1]
        #     fast_ema_latest_alt_rank = fast_ema_alt_rank.iloc[-1]
        #     slow_ema_latest_posts_created = slow_ema_posts_created.iloc[-1]
        #     fast_ema_latest_posts_created = fast_ema_posts_created.iloc[-1]
        #     slow_ema_latest_posts_active = slow_ema_posts_active.iloc[-1]
        #     fast_ema_latest_posts_active = fast_ema_posts_active.iloc[-1]
        #     slow_ema_latest_social_dominance = slow_ema_social_dominance.iloc[-1]
        #     fast_ema_latest_social_dominance = fast_ema_social_dominance.iloc[-1]
        #     slow_ema_latest_contributors_active = slow_ema_contributors_active.iloc[-1]
        #     fast_ema_latest_contributors_active = fast_ema_contributors_active.iloc[-1]
        #     slow_ema_latest_contributors_created = slow_ema_contributors_created.iloc[-1]
        #     fast_ema_latest_contributors_created = fast_ema_contributors_created.iloc[-1]
        #     slow_ema_latest_volatility = slow_ema_volatility.iloc[-1]
        #     fast_ema_latest_volatility = fast_ema_volatility.iloc[-1]
        # except:
        #     slow_ema_latest_sentiment=None
        #     fast_ema_latest_sentiment=None
        #     slow_ema_latest_alt_rank = None
        #     fast_ema_latest_alt_rank = None
        #     slow_ema_latest_posts_created = None
        #     fast_ema_latest_posts_created = None
        #     slow_ema_latest_posts_active = None
        #     fast_ema_latest_posts_active = None
        #     slow_ema_latest_social_dominance = None
        #     fast_ema_latest_social_dominance = None
        #     slow_ema_latest_contributors_active = None
        #     fast_ema_latest_contributors_active = None
        #     slow_ema_latest_contributors_created = None
        #     fast_ema_latest_contributors_created = None
        #     slow_ema_latest_volatility = None
        #     fast_ema_latest_volatility = None

        social_data_indicators = get_social_data_analysis(pair, indicators)
        
        # Price Indicators
        sma = indicators.get_indicator_value('sma', pair=pair)
        rsi = indicators.get_indicator_value("rsi", pair=pair)

        # Volume Based Metrics 
        volume = indicators.get_price(column="volume", pair=pair)
        prev_volume = indicators.get_price(column="volume", pair=pair, index=-2)
        sma_volume = indicators.get_indicator_value('sma_volume', pair=pair)
        
        
        

        # regime_val = indicators.get_indicator_value("regime", pair=pair)  # Because the regime filter is calculated only daily, we allow some lookback
        # regime=None
        # if regime_val is None:
        #     regime = Regime.bull
        # else:
        #     regime = Regime(regime_val)  # Convert to enum for readability
        last_close_price = indicators.get_price(pair=pair)

         #
        # Visualisations
        #

        if input.is_visualisation_enabled():
            visualisation = state.visualisation  # Helper class to visualise strategy output

            # Draw bollinger price levels on the price cahrt
            # visualisation.plot_indicator(timestamp, f"Social Mentions {pair.base}", PlotKind.technical_indicator_detached, social_mentions, colour="purple",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB upper mentions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_upper_mentions_social, colour="darkblue", detached_overlay_name=f"Social Mentions {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB lower mentions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_lower_mentions_social, colour="darkblue", detached_overlay_name=f"Social Mentions {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB mid mentions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_mid_mentions_social, colour="blue", detached_overlay_name=f"Social Mentions {pair.base}",pair=pair)


            # visualisation.plot_indicator(timestamp, f"Social Interactions {pair.base}", PlotKind.technical_indicator_detached, interactions, colour="red",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB upper interactions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_upper_interactions_social, colour="darkblue", detached_overlay_name=f"Social Interactions {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB lower interactions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_lower_interactions_social, colour="darkblue", detached_overlay_name=f"Social Interactions {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB mid interactions {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_mid_interactions_social, colour="blue", detached_overlay_name=f"Social Interactions {pair.base}",pair=pair)


            # visualisation.plot_indicator(timestamp, f"Sentiment {pair.base}", PlotKind.technical_indicator_detached, sentiment, colour="green",pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow MA sentiment {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_sentiment, colour="darkblue", detached_overlay_name=f"Sentiment {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"Fast MA sentiment {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_sentiment, colour="darkblue", detached_overlay_name=f"Sentiment {pair.base}",pair=pair)


            # Plotting the latest values for each metric in their respective detached chart names
            # visualisation.plot_indicator(timestamp, f"Slow EMA Alt Rank {pair.base}", PlotKind.technical_indicator_detached, slow_ema_latest_alt_rank, colour="green", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Fast EMA Alt Rank {pair.base}", PlotKind.technical_indicator_detached, fast_ema_latest_alt_rank, colour="green", pair=pair)

            visualisation.plot_indicator(timestamp, f"Posts Created {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['posts_created'], colour="orange", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow EMA Posts Created {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_posts_created, colour="green", pair=pair, detached_overlay_name=f"Posts Created {pair.base}")
            # visualisation.plot_indicator(timestamp, f"Fast EMA Posts Created {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_posts_created, colour="red", pair=pair, detached_overlay_name=f"Posts Created {pair.base}")

            # visualisation.plot_indicator(timestamp, f"Posts Active {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['posts_active'], colour="brown", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow EMA Posts Active {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_posts_active, colour="green", pair=pair, detached_overlay_name=f"Posts Active {pair.base}")
            # visualisation.plot_indicator(timestamp, f"Fast EMA Posts Active {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_posts_active, colour="green", pair=pair, detached_overlay_name=f"Posts Active {pair.base}")

            # visualisation.plot_indicator(timestamp, f"Social Dominance {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['social_dominance'], colour="blue", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow EMA Social Dominance {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_social_dominance, colour="green", pair=pair, detached_overlay_name=f"Social Dominance {pair.base}")
            # visualisation.plot_indicator(timestamp, f"Fast EMA Social Dominance {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_social_dominance, colour="green", pair=pair, detached_overlay_name=f"Social Dominance {pair.base}")

            # visualisation.plot_indicator(timestamp, f"Contributors Active {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['contributors_active'], colour="red", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow EMA Contributors Active {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_contributors_active, colour="green", pair=pair, detached_overlay_name=f"Contributors Active {pair.base}")
            # visualisation.plot_indicator(timestamp, f"Fast EMA Contributors Active {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_contributors_active, colour="green", pair=pair, detached_overlay_name=f"Contributors Active {pair.base}")

            visualisation.plot_indicator(timestamp, f"Contributors Created {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['contributors_created'], colour="blue", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Slow EMA Contributors Created {pair.base}", PlotKind.technical_indicator_overlay_on_detached, slow_ema_latest_contributors_created, colour="green", pair=pair, detached_overlay_name=f"Contributors Created {pair.base}")
            # visualisation.plot_indicator(timestamp, f"Fast EMA Contributors Created {pair.base}", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest_contributors_created, colour="red", pair=pair, detached_overlay_name=f"Contributors Created {pair.base}")

            # visualisation.plot_indicator(timestamp, f"Slow EMA Volatility {pair.base}", PlotKind.technical_indicator_detached, slow_ema_latest_volatility, colour="green", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Fast EMA Volatility {pair.base}", PlotKind.technical_indicator_detached, fast_ema_latest_volatility, colour="green", pair=pair)




            # visualisation.plot_indicator(timestamp, f"BB upper sentiment {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_upper_sentiment_social, colour="darkblue", detached_overlay_name=f"Sentiment {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB lower sentiment {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_lower_sentiment_social, colour="darkblue", detached_overlay_name=f"Sentiment {pair.base}",pair=pair)
            # visualisation.plot_indicator(timestamp, f"BB mid sentiment {pair.base}", PlotKind.technical_indicator_overlay_on_detached, bb_mid_sentiment_social, colour="blue", detached_overlay_name=f"Sentiment {pair.base}",pair=pair)

            # visualisation.plot_indicator(timestamp, f"Alt Rank {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['alt_rank'], colour="purple", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Social Ratio {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['ratio_increasing'], colour="black", pair=pair)
            # visualisation.plot_indicator(timestamp, f"Volatility {pair.base}", PlotKind.technical_indicator_detached, social_data_indicators['volatility'], colour="black", pair=pair)

            # Draw the RSI indicator on a separate chart pane.
            # Visualise the high RSI threshold we must exceed to take a position.
            visualisation.plot_indicator(timestamp, f"SMA {pair.base}", PlotKind.technical_indicator_on_price, sma, pair=pair)
            # visualisation.plot_indicator(timestamp, f"Volume {pair.base}", PlotKind.technical_indicator_detached, volume, pair=pair)
            # visualisation.plot_indicator(timestamp, f"Volume SMA {pair.base}", PlotKind.technical_indicator_overlay_on_detached, sma_volume, pair=pair, detached_overlay_name=f"Volume {pair.base}")
     



        
        # Check if we are too early in the backtesting to have enough data to calculate indicators
        # if None in (volume, bb_upper_interactions_social, bb_upper_interactions_social, bb_upper_sentiment_social, social_mentions, interactions, sentiment, sma):
        if None in (volume, social_data_indicators['interactions'], social_data_indicators['sentiment'], social_data_indicators['posts_created'], social_data_indicators['posts_active'], social_data_indicators['social_dominance'], social_data_indicators['contributors_active'], social_data_indicators['contributors_created']):
            continue

        # Check if the volume is increasing
        # volume_increasing = None
        # if volume is not None and is_volume_increasing(volume, prev_volume):
        #     # Volume is increasing, consider this in your trading decision
        #     # For example, set a flag or directly decide to trade based on this and other indicators
        #     volume_increasing = True
        # else:
        #     volume_increasing = False

        # Make sure you don't trade the same base token in current traded positions
        open_positions = state.portfolio.open_positions.values()
        base_token_address = pair.base.address
        quote_token_address = pair.quote.address
        # Check if there's already an open position with the same quote token
        existing_position_with_quote = any(
            pos.pair.base.address == base_token_address for pos in open_positions
        )

        # If there's already an open position with the same quote token, skip this pair
        if existing_position_with_quote:
            continue

        # crossover_mentions, mentions_cross_over_index = contains_cross_over(
        #     social_mentions_series,
        #     bb_upper_social_series, 
        #     lookback_period=2,
        #     must_return_index=True
        # )

        closed_positions = position_manager.get_closed_positions_for_pair(pair=pair)
        # print(f"Closed positions for pair {pair}:  {closed_positions[-1]}")

        if len(state.portfolio.open_positions) <= parameters.max_assets and state.portfolio.get_open_position_for_pair(pair) is None:
            # No open positions, decide if BUY in this cycle.
            # We buy if the price on the daily chart closes above the upper Bollinger Band.
            # if last_close_price > sma and crossover_mentions and mentions_cross_over_index == -1: #and rsi < 85 and rsi > 50:# and volume_increasing: # and volume > sma_volume:# and regime == Regime.bull:
            # if last_close_price > sma and sentiment > bb_mid_sentiment_social and (social_interactions > bb_upper_interactions_social or (crossover_mentions and mentions_cross_over_index == -1)):
            #     print(f"Placing a trade in for {pair.base} and voluem is {volume}, volume increase: {volume_increasing} and the rsi is {rsi}")
            #     buy_amount = cash * parameters.allocation
            #     # last_close_price
            #     # sma / last_close_price
            #     print(f"Sma right now is {sma} and new stop loss would be {sma / last_close_price} while current  price is {last_close_price}")
            #     trades += position_manager.open_spot(
            #             pair,
            #             value=buy_amount,
            #             stop_loss_pct=parameters.stop_loss_pct #sma / last_close_price,             
            #         )
            #     # Check for open condition - is the price breaking out
            # if rsi > 90:
            #     print(f"Closing Position due to high RSI", pair.base)
            #     position = position_manager.get_current_position_for_pair(pair)
            #     if position:
            #         trades += position_manager.close_position(position)

            # if social_data_indicators['social_interactions_ema_cross_over'] and social_data_indicators['sentiment_ema_cross_over'] and social_data_indicators['posts_created_ema_cross_over']:
            #     print(f"Placing a trade in for {pair.base} and voluem is {volume}, volume increase: {volume_increasing} and the rsi is {rsi}")
            #     buy_amount = cash * parameters.allocation
            #     # last_close_price
            #     # sma / last_close_price
            #     print(f"Sma right now is {sma} and new stop loss would be {sma / last_close_price} while current  price is {last_close_price}")
            #     trades += position_manager.open_spot(
            #             pair,
            #             value=buy_amount,
            #             stop_loss_pct=parameters.stop_loss_pct #sma / last_close_price,             
            #         )
            #     # Check for open condition - is the price breaking out
            # if rsi > 90:
            #     print(f"Closing Position due to high RSI", pair.base)
            #     position = position_manager.get_current_position_for_pair(pair)
            #     if position:
            #         trades += position_manager.close_position(position)

            """
            want to be able to do the block above but with different conditions depending on the array of strings I will pass as a parameter. 
            For example if my array is ['socialInc_dominanceInc_sentimentInc','socialInc_sentimentInc', 'socialInc_dominanceInc_sentimentInc_socialInteractionInc' ]
            The first one would use social_mentions_increase, and sentiment increase and social dominance increase.
            the second one would use social mentions increase and sentiment increase 
            and so on
            so I can iterate through all the combinations of triggers to see what works best
            if social_data_indicators['social_interactions_increasing'] and social_data_indicators['sentiment_increasing'] and social_data_indicators['social_dominance_increasing']: 
                print(f"Placing a trade in for {pair.base} and voluem is {volume}, volume increase: {volume_increasing} and the rsi is {rsi}")
                buy_amount = cash * parameters.allocation
                # last_close_price
                # sma / last_close_price
                print(f"Sma right now is {sma} and new stop loss would be {sma / last_close_price} while current  price is {last_close_price}")
                trades += position_manager.open_spot(
                        pair,
                        value=buy_amount,
                        stop_loss_pct=parameters.stop_loss_pct #sma / last_close_price,             
                    )
            """
            trigger_condition_set = parameters.trigger_conditions

            conditions_to_check = decode_conditions(trigger_condition_set)
            if check_conditions(trigger_condition_set, social_data_indicators):
                buy_amount = cash * parameters.allocation
                trades += position_manager.open_spot(
                        pair,
                        value=buy_amount,
                        stop_loss_pct=parameters.stop_loss_pct            
                    )
            


    
   

    return trades  # Return the list of trades we made in this cycle


# In[28]:


# from itertools import combinations

# # Define your single conditions
# single_conditions = [
#     'social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 
#     'posts_created_increasing', 'posts_active_increasing', 'social_dominance_increasing', 
#     'contributors_active_increasing', 'contributors_created_increasing',
#     'social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 
#     'posts_created_ema_cross_over', 'posts_active_ema_cross_over', 'social_dominance_ema_cross_over', 
#     'contributors_active_ema_cross_over', 'contributors_created_ema_cross_over'
# ]

# # Generate all combinations of up to 3 conditions
# trigger_conditions = [list(combo) for r in range(1, 4) for combo in combinations(single_conditions, r)]

# # Encode each combination
# encoded_conditions = ['&'.join(cond) for cond in trigger_conditions]

# # Example of limiting the encoded strings' lengths
# print("Number of combinations:", len(encoded_conditions))
# for encoded in encoded_conditions[:20]:  # Just display the first 5 to check
#     print(encoded)


# In[14]:


# def encode_conditions(conditions):
#     # Joins the list into a single string with '&' as a delimiter
#     return '&&'.join(conditions)

# trigger_conditions = [
#     ['social_mentions_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 'posts_created_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 'posts_created_increasing', 'posts_active_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 'posts_created_increasing', 'posts_active_increasing', 'social_dominance_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 'posts_created_increasing', 'posts_active_increasing', 'social_dominance_increasing', 'contributors_active_increasing'],
#     ['social_mentions_increasing', 'interactions_increasing', 'sentiment_increasing', 'posts_created_increasing', 'posts_active_increasing', 'social_dominance_increasing', 'contributors_active_increasing', 'contributors_created_increasing'],

#     ['social_mentions_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 'posts_created_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 'posts_created_ema_cross_over', 'posts_active_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 'posts_created_ema_cross_over', 'posts_active_ema_cross_over', 'social_dominance_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 'posts_created_ema_cross_over', 'posts_active_ema_cross_over', 'social_dominance_ema_cross_over', 'contributors_active_ema_cross_over'],
#     ['social_mentions_ema_cross_over', 'interactions_ema_cross_over', 'sentiment_ema_cross_over', 'posts_created_ema_cross_over', 'posts_active_ema_cross_over', 'social_dominance_ema_cross_over', 'contributors_active_ema_cross_over', 'contributors_created_ema_cross_over']
# ]

# # Encode each condition set to a string
# encoded_conditions = [encode_conditions(conditions) for conditions in trigger_conditions]

# Now 'encoded_conditions' contains each set of conditions as a single string, using '&' as a delimiter


# In[15]:




# In[20]:


# def check_conditions(conditions, indicators):
#     # Check if all conditions in the list are True in the social_data_indicators
#     return all(indicators.get(condition, False) for condition in conditions)

# # Example usage


# social_data_indicators = {'social_mentions': None, 'social_mentions_ema_3d': None, 'social_mentions_ema_6d': None, 'social_mentions_increasing': True, 'social_mentions_ema_cross_over': False, 'social_interactions': None, 'social_interactions_ema_3d': None, 'social_interactions_ema_6d': None, 'interactions_increasing': True, 'social_interactions_ema_cross_over': False, 'sentiment': None, 'sentiment_ema_3d': None, 'sentiment_ema_6d': None, 'sentiment_increasing': True, 'sentiment_ema_cross_over': False, 'posts_created': None, 'posts_created_ema_3d': None, 'posts_created_ema_6d': None, 'posts_created_increasing': True, 'posts_created_ema_cross_over': False, 'posts_active': None, 'posts_active_ema_3d': None, 'posts_active_ema_6d': None, 'posts_active_increasing': True, 'posts_active_ema_cross_over': False, 'social_dominance': None, 'social_dominance_ema_3d': None, 'social_dominance_ema_6d': None, 'social_dominance_increasing': False, 'social_dominance_ema_cross_over': False, 'contributors_active': None, 'contributors_active_ema_3d': None, 'contributors_active_ema_6d': None, 'contributors_active_increasing': False, 'contributors_active_ema_cross_over': False, 'contributors_created': None, 'contributors_created_ema_3d': None, 'contributors_created_ema_6d': None, 'contributors_created_increasing': False, 'contributors_created_ema_cross_over': False}

# def decode_conditions(encoded_string):
#     # Splits the string back into the list of conditions using '&' as the delimiter
#     return encoded_string.split('&&')


# for condition_set in encoded_conditions:
#     conditions_to_check = decode_conditions(condition_set)
#     print('conditions_to_check', conditions_to_check)
#     print('check', check_conditions(conditions_to_check, social_data_indicators))
#     if check_conditions(conditions_to_check, social_data_indicators):
#         print(f"Conditions met for: {condition_set}")


# In[ ]:





# # Backtest
# 
# - Run the backtest

# In[57]:


# import logging
# from tradeexecutor.backtest.backtest_runner import run_backtest_inline

# try:
#     result = run_backtest_inline(
#         name=parameters.id,
#         engine_version="0.5",
#         decide_trades=decide_trades,
#         create_indicators=create_indicators,
#         client=client,
#         universe=strategy_universe,
#         parameters=parameters,
#         strategy_logging=False,
#         max_workers=10,
#         reserve_currency=parameters.RESERVE_CURRENCY,  # USDC.e bridged on Polygon
#         # We need to set this really high value, because
#         # some low cap tokens may only see 1-2 trades per year
#         # and our backtesting framework aborts if it thinks
#         # there is an issue with data quality
#         data_delay_tolerance=pd.Timedelta(days=365),
#         minimum_data_lookback_range=pd.Timedelta(days=40+1)

#         # Uncomment to enable verbose logging
#         # log_level=logging.INFO,
#     )
#     state = result.state
#     trade_count = len(list(state.portfolio.get_all_trades()))
#     print(f"Backtesting completed, backtested strategy made {trade_count} trades")
#     print(state.portfolio.get_all_trades())
# except Exception as e:
#     print("error", e)
#     print(e.__cause__)
#     raise e





# In[ ]:


# print(f"All trades for this backtest: {result.state.portfolio.get_all_trades()}")


# # Indicator diagnostics
# 
# - Diagnose the custom data we use
# 

# # In[61]:


# indicators = result.indicators

# indicator_name = "social_mentions"

# for pair in strategy_universe.iterate_pairs():
#     data = indicators.get_indicator_series(indicator_name, pair=pair, unlimited=True)
#     if data is None or data.empty:
#         print(f"FAIL {pair}: No {indicator_name} data")
#     elif (data.fillna(0) == 0).all():
#         print(f"FAIL {pair}: {indicator_name} data all zeroes/NA")
#     else:
#         # Add debug printing if needed
#         print(f"OK {pair}: {indicator_name} contains {len(data)} samples")
        # pass


# - Visualise some custom indicator data, so we know it looks correct
# - We pick one of tokens in the sample data set, and pull outs it custom indicator data generated or loaded during the backtest run

# # In[65]:


# trading_pairs = strategy_universe.data_universe.pairs
# trading_pairs.get_pair('pendle')


# # In[21]:

# dd
# pairs_dict = {}
# for pair in strategy_universe.iterate_pairs():
#     print(f"Pair : {pair.internal_id} with the following description {pair.get_human_description()} and {pair.get_human_description()}")
#     pairs_dict[pair.get_identifier] = pair


# In[68]:


strategy_universe.get_trading_pair(2955795)


# # In[124]:


# from tradeexecutor.visual.single_pair import visualise_single_pair
# # from tradeexecutor.visual.
# from tradingstrategy.charting.candle_chart import VolumeBarMode

# start_at, end_at = state.get_strategy_start_and_end()   # Limit chart to our backtesting range
# # btc_usdt = strategy_universe.get_pair_by_human_description(trading_pairs[0])
# # sample_pair = strategy_universe.get_pair_by_human_description((ChainId.arbitrum, "uniswap-v3", "PENDLE", "WETH", 0.003))
# sample_pair = strategy_universe.get_trading_pair(2955795)


# figure = visualise_single_pair(
#     state,
#     pair_id=sample_pair.internal_id,
#     execution_context=notebook_execution_context,
#     candle_universe=strategy_universe.data_universe.candles,
#     # start_at=start_at,
#     # end_at=end_at,
#     vertical_spacing=0.045,
#     volume_bar_mode=VolumeBarMode.hidden,
#     volume_axis_name="Volume (USD)",
#     height = 800,
#     title="PENDLE trades",
#     detached_indicators=True
# )

# figure.show()


# Show raw custom indicator data.

# In[10]:


# display(sentiment_series.iloc[0:10])


# # Equity curve
# 
# - Equity curve shows how your strategy accrues value over time
# - A good equity curve has a stable ascending angle
# - Benchmark against MATIC buy and hold

# # In[121]:


# import pandas as pd
# from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
# from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark

# # Pulls WMATIC/USDC as the benchmark
# benchmark_indexes = get_benchmark_data(
#     strategy_universe,
#     cumulative_with_initial_cash=state.portfolio.get_initial_cash()
# )

# fig = visualise_equity_curve_benchmark(
#     name=state.name,
#     portfolio_statistics=state.stats.portfolio,
#     all_cash=state.portfolio.get_initial_cash(),
#     benchmark_indexes=benchmark_indexes,
#     height=800,
#     log_y=True,
# )

# fig.show()


# # Performance metrics
# 
# - Display portfolio performance metrics
# - Compare against buy and hold matic using the same initial capital

# In[122]:


# from tradeexecutor.analysis.multi_asset_benchmark import compare_strategy_backtest_to_multiple_assets

# compare_strategy_backtest_to_multiple_assets(
#     state,
#     strategy_universe,
#     display=True,
# )


# # # Trading statistics
# 
# - Display summare about made trades

# In[123]:


# from tradeexecutor.analysis.trade_analyser import build_trade_analysis

# analysis = build_trade_analysis(state.portfolio)
# summary = analysis.calculate_summary_statistics()
# display(summary.to_dataframe())


# # Pair breakdown
# 
# - Profit for each trading pair

# # In[76]:


# from tradeexecutor.analysis.multipair import analyse_multipair
# from tradeexecutor.analysis.multipair import format_multipair_summary

# multipair_summary = analyse_multipair(state)
# display(format_multipair_summary(multipair_summary))


# In[ ]:





# # Grid search
# 
# - Run the grid search

# In[50]:


from tradeexecutor.backtest.grid_search import GridCombination, get_grid_search_result_path, perform_grid_search, prepare_grid_combinations

# This is the path where we keep the result files around
storage_folder = get_grid_search_result_path(Parameters.id)

# Popular grid search combinations and indicators for them
combinations = prepare_grid_combinations(
    parameters,
    storage_folder,
    create_indicators=create_indicators,
    strategy_universe=strategy_universe,
)

indicators = GridCombination.get_all_indicators(combinations)

print(f"We prepared {len(combinations)} grid search combinations with total {len(indicators)} indicators which need to be calculated, stored in {storage_folder.resolve()}")

grid_search_results = perform_grid_search(
    decide_trades,
    strategy_universe,
    combinations,
    trading_strategy_engine_version="0.5",
    multiprocess=True,
)


# In[96]:


grid_search_results


# ### Analyse all results

# In[97]:


from tradeexecutor.analysis.grid_search import analyse_grid_search_result
from tradeexecutor.analysis.grid_search import render_grid_search_result_table

df = analyse_grid_search_result(grid_search_results)
print(f"We have {len(df)} results")
render_grid_search_result_table(df)


# ### Highest Profitability

# In[98]:


from tradeexecutor.analysis.grid_search import find_best_grid_search_results, render_grid_search_result_table

best_result = find_best_grid_search_results(grid_search_results)
render_grid_search_result_table(best_result.cagr)


# ### Highest Sharpe Ratio

# In[145]:


render_grid_search_result_table(best_result.sharpe)


# # Alpha model thinking
# 
# - Show the portfolio construction steps for each decision cycle
# - Render a table where we show what was the signal for each asset and how much it was bought or sod

# In[15]:


# from IPython.display import HTML

# from tradeexecutor.analysis.alpha_model_analyser import render_alpha_model_plotly_table, create_alpha_model_timeline_all_assets

# # Render alpha model timeline as Pandas HTML table
# timeline = create_alpha_model_timeline_all_assets(state, strategy_universe, new_line="CcC")
# HTML(timeline.to_html().replace("CcC", "<br>"))


# Alternative Plotly renderer,
# does not work in notebook HTML export but has richer output
#figure, table = render_alpha_model_plotly_table(timeline)

