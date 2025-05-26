"""Memecoin strategy based on market sentiment. Long only.

To backtest this strategy module locally:

.. code-block:: console

    source scripts/set-latest-tag-gcp.sh
    docker-compose run ethereum-memecoin-swing backtest

Or:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategy/ethereum-memecoin-swing.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

    trade-executor \
    backtest \
    --strategy-file=strategies/prod_strategies/ethereum-memecoin-swing.py \
    --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

"""
import datetime
import json
import logging
import os
from pathlib import Path

import pandas as pd
import pandas_ta

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.alternative_market_data import resample_multi_pair
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.strategy_input import (
    IndicatorDataNotFoundWithinDataTolerance,
    StrategyInput,
    StrategyInputIndicators,
)
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
    translate_token,
)
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.crossover import contains_cross_over, contains_cross_under

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import (
    HumanReadableTradingPairDescription,
    PandasPairUniverse,
    StablecoinFilteringMode,
    filter_for_base_tokens,
    filter_for_stablecoins,
)
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.utils.token_filter import deduplicate_pairs_by_volume, add_base_quote_address_columns
from tradingstrategy.utils.token_extra_data import load_extra_metadata


logger = logging.getLogger(__name__)

trading_strategy_engine_version = "0.5"


from tradingstrategy.chain import ChainId
import datetime

from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.parameters import StrategyParameters


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "eth-memecoins" # Used in cache paths

    cycle_duration = CycleDuration.d1  # Daily rebalance
    candle_time_bucket = TimeBucket.d1  
    allocation = 0.142
    max_assets = 7  

    #
    # Liquidity risk analysis and data quality
    #
    min_price = 0.00000000000000000001  
    max_price = 1_000_000  
    min_liquidity_trade_threshold = 0.05
    min_liquidity_threshold = 25000 
    min_volume = 30000 

    #Trigger
    #Safety Guards
    minimum_mometum_threshold = 0.1 
    momentum_lookback_bars = 9

    sma_length = 12 
    social_ema_short_length = 6
    social_ema_long_length = 11
    cross_over_period = 2
    social_ma_min = 10

    stop_loss_pct = 0.95
    trailing_stop_loss_pct = 0.86
    trailing_stop_loss_activation_level = 1.41

    # Trade execution parameters
    slippage_tolerance = 0.06
    max_buy_tax = 0.06
    max_sell_tax = 0.06
    token_risk_threshold = 50
    # If the pair does not have enough real time quote token TVL, skip trades smaller than this
    min_trade_size_usd = 5.00
    # Only do trades where we are less than 1% of the pool quote token TVL
    per_position_cap_of_pool = 0.01
    
    #
    # Live trading only
    #
    chain_id = ChainId.base
    routing = TradeRouting.default  
    required_history_period = datetime.timedelta(days=30 + 1)

    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2024, 12, 1)
    backtest_end = datetime.datetime(2024, 12, 31)
    initial_cash = 10_000
    initial_deposit = 10_000

    stop_loss_time_bucket = TimeBucket.h1


parameters = StrategyParameters.from_class(Parameters)  


def join_market_trend_data(df_trend, market_data):
    df_trend.reset_index(inplace=True)
    market_data.reset_index(inplace=True)

    # Ensure "date" in df_trend_new is also in datetime format (if not already done)
    df_trend["date"] = pd.to_datetime(df_trend["date"])
    market_data["date"] = pd.to_datetime(market_data["date"])
    market_data["date"] = market_data["date"].dt.tz_localize(None)

    # Perform the merge
    merged_df = pd.merge(df_trend, market_data, how="left", on=["coin_id", "date"])

    merged_df.set_index(["coin_id", "coin", "date"], inplace=True)

    return merged_df


def get_google_storage_credentials():
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_dict = json.loads(credentials_json)

    return {"token": credentials_dict}


def get_trend_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    storage_options = get_google_storage_credentials()

    df_platforms = pd.read_csv(
        "gs://taraxa-research/coingecko_data_pipeline/df_platforms.csv",
        storage_options=storage_options,
    )

    df_categories = pd.read_csv(
        "gs://taraxa-research/coingecko_data_pipeline/df_categories.csv",
        storage_options=storage_options,
    )

    df_stablecoins = df_categories[df_categories["category_name"] == "Stablecoins"]

    df_markets_data = pd.read_parquet(
        "gs://taraxa-research/coingecko_data_pipeline/df_market_data_historical.parquet",
        storage_options=storage_options,
    )
    df_trend = pd.read_parquet(
        "gs://taraxa-research/trend_spotting/trends_v1.parquet",
        storage_options=storage_options,
    )
    df_trend.reset_index(inplace=True)
    df_trend_eth = df_trend.merge(
        df_platforms[df_platforms["platform_name"] == "Base"],
        on="coin_id",
        how="inner",
    )
    df_trend = df_trend_eth.merge(
        df_categories[df_categories["category_name"] == "Meme"],
        on="coin_id",
        how="inner",
    )

    df_trend_w_market = join_market_trend_data(df_trend, df_markets_data)
    df_trend_w_market.reset_index(inplace=True)
    df_trend_w_market = df_trend_w_market.rename(
        columns={"coin": "symbol", "coin_id_count": "social_mentions"}
    )
    df_trend_eth = df_trend_eth.rename(
        columns={"coin": "symbol", "coin_id_count": "social_mentions"}
    )

    return df_trend_w_market, df_trend_eth, df_stablecoins

def filter_pairs_by_risk(
    pairs_df: pd.DataFrame,
    risk_threshold: int = 60,
    max_buy_tax: float = 6.0,
    max_sell_tax: float = 6.0,
    risk_traits: dict = None,
) -> pd.DataFrame:
    """Filter pairs DataFrame based on tax rates, TokenSniffer risk score, and specific risk traits.

    Args:
        pairs_df (pd.DataFrame): DataFrame containing trading pair information
        risk_threshold (int): Minimum acceptable TokenSniffer risk score (0-100, higher is better)
        max_buy_tax (float): Maximum allowed buy tax percentage (default 6.0)
        max_sell_tax (float): Maximum allowed sell tax percentage (default 6.0)
        risk_traits (dict): Dictionary of risk traits to filter on. If None, only tax and risk score are checked

    Returns:
        pd.DataFrame: Filtered pairs DataFrame containing only pairs meeting all criteria

    Example Risk Traits Dictionary:
    ```python
    # Complete risk traits dictionary
    risk_traits = {
        # Contract-level risks
        'has_mint': False,                    # Can new tokens be minted
        'has_fee_modifier': False,            # Can fees be modified after deployment
        'has_max_transaction_amount': False,   # Presence of max transaction limits
        'has_blocklist': False,               # Can addresses be blacklisted
        'has_proxy': False,                   # Is it a proxy contract (upgradeable)
        'has_pausable': False,                # Can trading be paused

        # Ownership and control risks
        'is_ownership_renounced': True,       # Ownership should be renounced
        'is_source_verified': True,           # Contract should be verified

        # Trading risks
        'is_sellable': True,                  # Token can be sold
        'has_high_buy_fee': False,            # High buy fees present
        'has_high_sell_fee': False,           # High sell fees present
        'has_extreme_fee': False,             # Extremely high fees (>30%)

        # Liquidity risks
        'has_inadequate_liquidity': False,    # Insufficient liquidity
        'has_inadequate_initial_liquidity': False,  # Started with low liquidity

        # Token distribution risks
        'has_high_creator_balance': False,    # Creator holds large portion
        'has_high_owner_balance': False,      # Owner holds large portion
        'has_high_wallet_balance': False,     # Any wallet holds too much
        'has_burned_exceeds_supply': False,   # Burned amount > supply (impossible)

        # Additional safety checks
        'is_flagged': False,                  # Token is flagged for issues
        'is_honeypot': False,                 # Known honeypot
        'has_restore_ownership': False,       # Can ownership be restored
        'has_non_standard_erc20': False       # Non-standard ERC20 implementation
    }
    ```

    Example Risk Profiles:
    ```python
    # Conservative (strict) settings
    conservative_risk_traits = {
        'has_mint': False,
        'has_fee_modifier': False,
        'has_blocklist': False,
        'has_proxy': False,
        'has_pausable': False,
        'is_ownership_renounced': True,
        'is_source_verified': True,
        'is_sellable': True,
        'has_high_buy_fee': False,
        'has_high_sell_fee': False,
        'is_flagged': False,
        'is_honeypot': False
    }

    # Moderate settings
    moderate_risk_traits = {
        'has_mint': False,
        'is_source_verified': True,
        'is_sellable': True,
        'has_extreme_fee': False,
        'is_honeypot': False,
        'is_flagged': False
    }

    # Aggressive settings
    aggressive_risk_traits = {
        'is_sellable': True,
        'is_honeypot': False,
        'is_flagged': False
    }
    ```

    Usage:
    ```python
    # Using conservative settings with custom tax limits
    filtered_df = filter_pairs_by_risk(
        pairs_df,
        risk_threshold=60,
        max_buy_tax=5.0,
        max_sell_tax=5.0,
        risk_traits=conservative_risk_traits
    )

    # Custom risk profile
    custom_risk_traits = {
        'is_sellable': True,
        'is_honeypot': False,
        'has_mint': False,
        'has_extreme_fee': False,
        'is_source_verified': True
    }
    filtered_df = filter_pairs_by_risk(
        pairs_df,
        risk_threshold=70,
        max_buy_tax=3.0,
        max_sell_tax=3.0,
        risk_traits=custom_risk_traits
    )
    ```
    """
    # Create a copy to avoid modifying original
    filtered_df = pairs_df.copy()
    initial_count = len(filtered_df)

    # Replace NaN values with 0 for buy_tax and sell_tax
    filtered_df["buy_tax"] = filtered_df["buy_tax"].fillna(0)
    filtered_df["sell_tax"] = filtered_df["sell_tax"].fillna(0)

    # Filter for pairs meeting tax thresholds
    filtered_df = filtered_df[
        (filtered_df["buy_tax"] <= max_buy_tax)
        & (filtered_df["sell_tax"] <= max_sell_tax)
    ]

    after_tax_count = len(filtered_df)
    print(f"After tax filter we have {after_tax_count} trading pairs")

    def check_token_risk(row):
        try:
            # Extract TokenSniffer data from the nested structure
            token_data = row["other_data"]["top_pair_data"].token_sniffer_data
            if token_data is None:
                return False

            print(f"Token data: {token_data}")
            # Check risk score threshold
            if token_data.get("riskScore", 0) < risk_threshold:
                return False

            # Check each specified risk trait if provided
            if risk_traits:
                for trait, desired_value in risk_traits.items():
                    if token_data.get(trait, not desired_value) != desired_value:
                        return False

            return True

        except (KeyError, AttributeError) as e:
            print(f"Error processing row: {e}")
            return False

    # Apply TokenSniffer filters if risk_traits provided
    if risk_traits is not None:
        filtered_df = filtered_df[filtered_df.apply(check_token_risk, axis=1)]

    final_count = len(filtered_df)

    print(
        "Filtering results: Initial pairs: %d, after tax filters: %d, after risk filters: %d",
        initial_count,
        after_tax_count,
        final_count,
    )

    return filtered_df

def prefilter_data(client: Client, execution_context) -> pd.DataFrame:
    # If the pair does not have this liquidity, skip
    min_prefilter_liquidity = 10_000

    chain_id = Parameters.chain_id
    time_bucket = Parameters.candle_time_bucket

    # We need pair metadata to know which pairs belong to Polygon
    logger.info("Downloading/opening pairs dataset")
    pairs_df = client.fetch_pair_universe().to_pandas()
    our_chain_pair_ids = pairs_df[pairs_df.chain_id == chain_id.value][
        "pair_id"
    ].unique()

    logger.info(
        f"We have data for {len(our_chain_pair_ids)} trading pairs on {chain_id.name}"
    )

    # Download all liquidity data, extract
    # trading pairs that exceed our prefiltering threshold
    logger.info("Downloading/opening liquidity dataset")
    liquidity_df = client.fetch_all_liquidity_samples(time_bucket).to_pandas()
    logger.info(f"Filtering out liquidity for chain {chain_id.name}")
    liquidity_df = liquidity_df.loc[liquidity_df.pair_id.isin(our_chain_pair_ids)]
    liquidity_df = liquidity_df.loc[
        liquidity_df.timestamp > parameters.backtest_start
    ]
    liquidity_per_pair = liquidity_df.groupby(liquidity_df.pair_id)
    logger.info(
        f"Chain {chain_id.name} has liquidity data for {len(liquidity_per_pair.groups)}"
    )

    passed_pair_ids = set()
    liquidity_output_chunks = []

    for pair_id, pair_df in liquidity_per_pair:
        if pair_df["high"].max() > min_prefilter_liquidity:
            liquidity_output_chunks.append(pair_df)
            passed_pair_ids.add(pair_id)

    logger.info(
        f"After filtering for {min_prefilter_liquidity:,} USD min liquidity we have {len(passed_pair_ids)} pairs"
    )

    liquidity_df = pd.concat(liquidity_output_chunks)
    # liquidity_out_df.to_parquet(liquidity_output_fname)

    # logger.info(
    #     f"Wrote {liquidity_output_fname}, {liquidity_output_fname.stat().st_size:,} bytes"
    # )

    logger.info("Downloading/opening OHLCV dataset")
    price_df = client.fetch_all_candles(time_bucket).to_pandas()
    price_df = price_df.loc[price_df.pair_id.isin(passed_pair_ids)]
    # price_df.to_parquet(price_output_fname)

    # logger.info(
    #     f"Wrote {price_output_fname}, {price_output_fname.stat().st_size:,} bytes"
    # )

    # FILTER MORE

    NUM_TOKENS = 10

    custom_data_df, df_trend, df_stablecoins = get_trend_data()

    custom_data_df.index = pd.DatetimeIndex(custom_data_df["date"])
    custom_data_df["contract_address"] = custom_data_df[
        "contract_address"
    ].str.lower()  # Framework operates with lowercased addresses everywehre
    df_trend["contract_address"] = df_trend[
        "contract_address"
    ].str.lower()  # Framework operates with lowercased addresses everywehre
    custom_data_df.sort_index(ascending=True)
    custom_data_no_dups = custom_data_df.drop_duplicates(
        subset=["coin_id", "symbol"], keep="last"
    )[["coin_id", "symbol", "contract_address", "platform_name", "total_volume"]]

    start = custom_data_df.index[0]
    end = custom_data_df.index[-1]

    csv_token_list = list(custom_data_df.contract_address.unique())
    logger.info(
        f"CSV contains data for {len(csv_token_list)} tokens, time range {start} - {end}"
    )

    # Remove Stablecoins
    custom_data_no_dups = custom_data_no_dups[
        ~custom_data_no_dups.coin_id.isin(df_stablecoins.coin_id.unique())
    ]

    csv_token_list_backtest = custom_data_no_dups.sort_values(
        by="total_volume", ascending=False
    )[["symbol", "contract_address"]].iloc[:NUM_TOKENS]
    logger.info(
        f"Pre-selecting the following tokens and contract addresses to backtest {csv_token_list_backtest}"
    )

    base_erc20_address_list = []
    erc20_addresses_avoid = ["0xA3c322Ad15218fBFAEd26bA7f616249f7705D945".lower()]
    base_erc20_address_list += csv_token_list_backtest["contract_address"].tolist()
    base_erc20_address_list = [
        address
        for address in set(base_erc20_address_list)
        if address.lower() not in erc20_addresses_avoid
    ]

    # Move logic from create_trading_universe here
    SUPPORTED_DEXES = {"uniswap-v3", "uniswap-v2", "sushi"}

    # Get the token list of everything in the CSV + hardcoded WETH
    desired_trading_addresses = set(base_erc20_address_list)


    exchange_universe = client.fetch_exchange_universe()
    exchange_universe = exchange_universe.limit_to_chains(
        {Parameters.chain_id}
    ).limit_to_slugs(SUPPORTED_DEXES)

    pairs_df = client.fetch_pair_universe().to_pandas()

    logger.info(
        f"Prefilter data contains {len(liquidity_df):,} liquidity samples dn {len(price_df):,} OHLCV candles"
    )

    if (
        execution_context.live_trading
        or execution_context.mode == ExecutionMode.preflight_check
    ):
        # for live trading we only need the last 60 days
        now = datetime.datetime.utcnow()
        start_at = now - datetime.timedelta(days=60)
        end_at = now

        price_df = price_df.loc[price_df.timestamp >= start_at]
        liquidity_df = liquidity_df.loc[liquidity_df.timestamp >= start_at]

    elif execution_context.mode == ExecutionMode.backtesting:
        start_at = Parameters.backtest_start
        end_at = Parameters.backtest_end

        price_df = price_df.loc[
            (price_df.timestamp >= start_at) & (price_df.timestamp <= end_at)
        ]
        liquidity_df = liquidity_df.loc[
            (liquidity_df.timestamp >= start_at) & (liquidity_df.timestamp <= end_at)
        ]
    else:
        raise RuntimeError(f"Unknown execution mode {execution_context.mode}")

    # Prefilter for more liquidity conditions
    liquidity_per_pair = liquidity_df.groupby(liquidity_df.pair_id)
    logger.info(
        f"Chain {chain_id.name} has liquidity data for {len(liquidity_per_pair.groups)}"
    )

    # Get the date 30 days ago
    thirty_days_ago = end_at - datetime.timedelta(days=30)

    # Create a subset of the data for the last 30 days, in live trading this makes more sense
    liquidity_last_30_days = liquidity_df[liquidity_df.timestamp > thirty_days_ago]
    liquidity_per_pair_last_30d = liquidity_last_30_days.groupby("pair_id")
    passed_pair_ids = set()
    for pair_id, pair_df in liquidity_per_pair_last_30d:
        # Check the maximum high liquidity in the last 30 days
        if pair_df["high"].max() > Parameters.min_liquidity_threshold:
            passed_pair_ids.add(pair_id)

    pairs_df = pairs_df.loc[pairs_df.pair_id.isin(passed_pair_ids)]
    logger.info(
        f"After liquidity filter {Parameters.min_liquidity_threshold:,} USD we have {len(pairs_df)} trading pairs"
    )

    allowed_exchange_ids = set(exchange_universe.exchanges.keys())
    pairs_df = pairs_df.loc[pairs_df.exchange_id.isin(allowed_exchange_ids)]
    logger.info(f"After DEX filter we have {len(pairs_df)} trading pairs")

    # Store reference USDC ETH pair so we have an example pair with USDC as quote token for reserve asset
    eth_usdc_addresses = [
        "0x88a43bbdf9d098eec7bceda4e2494615dfd9bb9c",  # Uniswap V2,
        "0xd0b53d9277642d899df5c87a3966a349a798f224"  # Uniswap V3
    ]
    print(f"ETH-USDC pairs: {eth_usdc_addresses}")
    
    ref_usdc_pairs = pairs_df[
        pairs_df["address"].isin([addr.lower() for addr in eth_usdc_addresses])
    ].copy()

    # Pairs pre-processing
    pairs_df = add_base_quote_address_columns(pairs_df)
    pairs_df = pairs_df.loc[
        (pairs_df["base_token_address"].isin(desired_trading_addresses)) &
        (pairs_df["chain_id"] == chain_id)
    ]

    # Retrofit TokenSniffer data
    pairs_df = load_extra_metadata(
        pairs_df,
        client=client,
    )

    pairs_df = filter_pairs_by_risk(pairs_df, 
                        risk_threshold = Parameters.token_risk_threshold,
                        max_buy_tax = Parameters.max_buy_tax, 
                        max_sell_tax = Parameters.max_sell_tax,
                        risk_traits = None)

    # Do cross-section of tokens from custom data
    pairs_df = filter_for_stablecoins(
        pairs_df, StablecoinFilteringMode.only_volatile_pairs
    )
    logger.info(
        f"After custom data ERC-20 token address cross section filter we have {len(pairs_df)} trading pairs"
    )

    # Add back USDC pair that have ref pair in the pairs universe
    ref_usdc_pairs = ref_usdc_pairs[ref_usdc_pairs['pair_id'].isin(passed_pair_ids)]
    pairs_df = pd.concat([pairs_df, ref_usdc_pairs]).drop_duplicates(subset=['pair_id'])

    logger.info(f"Before deduplication we have {len(pairs_df)} trading pairs")
    pairs_df = deduplicate_pairs_by_volume(pairs_df)
    logger.info(f"After deduplication we have {len(pairs_df)} trading pairs")

    return pairs_df, price_df, liquidity_df, exchange_universe


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe."""

    USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913".lower()
    WETH = "0x4200000000000000000000000000000000000006".lower()

    pairs_df, price_df, liquidity_df, exchange_universe = prefilter_data(
        client, execution_context
    )

    if execution_context.mode == ExecutionMode.backtesting:
        # Resample strategy decision candles to daily
        daily_candles = resample_multi_pair(price_df, Parameters.candle_time_bucket)
        daily_candles["timestamp"] = daily_candles.index

        logger.info(
            f"After downsampling we have {len(daily_candles)} OHLCV candles and {len(liquidity_df)} liquidity samples"
        )
        candle_universe = GroupedCandleUniverse(
            daily_candles,
            time_bucket=Parameters.candle_time_bucket,
            forward_fill=True,  # Forward will should make sure we can always calculate RSI, other indicators
        )

        liquidity_universe = GroupedLiquidityUniverse(liquidity_df)

        # The final trading pair universe contains metadata only for pairs that passed
        # our filters
        pairs_universe = PandasPairUniverse(
            pairs_df,
            exchange_universe=exchange_universe,
        )
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

        reserve_asset = translate_token(pairs_universe.get_token(USDC))

        strategy_universe = TradingStrategyUniverse(
            data_universe=data_universe,
            backtest_stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            backtest_stop_loss_candles=stop_loss_candle_universe,
            reserve_assets=[reserve_asset],
        )

    elif execution_context.live_trading:

        dataset = load_partial_data(
            client,
            execution_context=execution_context,
            time_bucket=Parameters.candle_time_bucket,
            pairs=pairs_df,
            universe_options=universe_options,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            # lending_reserves=lending_reserves,
            required_history_period=Parameters.required_history_period,
            liquidity=True,
        )

        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            forward_fill=True,
            reserve_asset=USDC,
        )

    return strategy_universe


broken_trading_pairs = None


def get_broken_pairs(strategy_universe: TradingStrategyUniverse, parameters) -> set:
    # Run some extra sanity check for small cap tokens
    broken_trading_pairs = set()
    pairs_to_avoid = [87449]

    for pair in strategy_universe.iterate_pairs():
        reason = strategy_universe.get_trading_broken_reason(
            pair,
            min_candles_required=10,
            min_price=parameters.min_price,
            max_price=parameters.max_price,
        )
        if pair.internal_id in pairs_to_avoid:
            broken_trading_pairs.add(pair)
        if reason:
            logger.debug(
                f"FAIL: {pair} with base token {pair.base.address} may be problematic: {reason}"
            )
            broken_trading_pairs.add(pair)
        else:
            logger.debug(f"OK: {pair} included in the backtest")

    logger.info(
        f"Total {len(broken_trading_pairs)} broken trading pairs detected, having {strategy_universe.get_pair_count() - len(broken_trading_pairs)} good pairs left to trade"
    )

    return broken_trading_pairs


def is_acceptable(
    indicators: StrategyInputIndicators,
    parameters: StrategyParameters,
    pair: TradingPairIdentifier,
) -> bool:
    """Check the pair for risk acceptance

    :return:
        True if we should trade this pair
    """

    # TODO: maybe filter out directly when construct the universe?
    global broken_trading_pairs

    if broken_trading_pairs is None:
        broken_trading_pairs = get_broken_pairs(
            indicators.strategy_universe, parameters
        )

    if pair in broken_trading_pairs:
        # Don't even bother to try trade this
        return False

    avoid_backtesting_tokens = {
        # Trading jsut stops (though there is liq left)
        # https://tradingstrategy.ai/trading-view/ethereum/uniswap-v3/id-usdc-fee-30
        "PEOPLE",
        "WBTC",
    }

    if pair.base.token_symbol in avoid_backtesting_tokens:
        # Manually blacklisted toen for this backtest
        return False

    # Pair does not quality yet due to low liquidity
    liquidity = indicators.get_tvl(pair=pair)
    if liquidity is None or liquidity <= parameters.min_liquidity_threshold:
        return False

    volume = indicators.get_price(pair, column="volume")
    close_price = indicators.get_price(pair=pair)
    volume_adjusted = volume / close_price

    logger.info(f"Volume {volume} and fixed volume is {volume_adjusted}")
    if volume_adjusted < parameters.min_volume:
        return False

    return True


def get_custom_data_group():
    storage_options = get_google_storage_credentials()

    logger.info("Getting custom data group")

    df_platforms = pd.read_csv(
        "gs://taraxa-research/coingecko_data_pipeline/df_platforms.csv",
        storage_options=storage_options,
    )

    df_trend = pd.read_parquet(
        "gs://taraxa-research/trend_spotting/trends_v1.parquet",
        storage_options=storage_options,
    )
    df_trend.reset_index(inplace=True)
    df_trend_eth = df_trend.merge(
        df_platforms[df_platforms["platform_name"] == "Ethereum"],
        on="coin_id",
        how="inner",
    )

    df_trend_eth = df_trend_eth.rename(
        columns={"coin": "symbol", "coin_id_count": "social_mentions"}
    )

    df_trend_eth["contract_address"] = df_trend_eth[
        "contract_address"
    ].str.lower()  # Framework operates with lowercased addresses everywehre

    # Create per-pair DataFrame group by
    custom_data_group = (
        df_trend_eth.set_index("date").sort_index().groupby("contract_address")
    )

    return custom_data_group


custom_data_group = None


def add_metric(pair: TradingPairIdentifier, metric_name: str) -> pd.Series:
    """Add a specific metric to the dataset."""
    global custom_data_group

    if custom_data_group is None:
        custom_data_group = get_custom_data_group()

    try:
        contract_address = pair.base.address
        per_pair = custom_data_group.get_group(contract_address)
        series = per_pair[metric_name]

        # Check for duplicates
        duplicates = series.index.duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate entries for {pair} {metric_name}")
            
        metric_series = series[~series.index.duplicated(keep='last')].rename(metric_name)
        metric_series.sort_index(inplace=True)
        return metric_series
    except Exception as e:
        print(f"Error adding metric {metric_name} for pair {pair}: {str(e)}")
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def calculate_metric_emas(
    pair: TradingPairIdentifier, metric_name: str, short_length: int, long_length: int
) -> pd.DataFrame:
    """Calculate short and long EMAs for a specific metric."""
    metric_series = add_metric(pair, metric_name)
    metric_series = metric_series.interpolate(
        method="linear", limit_direction="forward"
    )
    ema_short = (
        metric_series.ewm(span=short_length, adjust=False)
        .mean()
        .rename(f"{metric_name}_ema_short")
    )
    ema_long = (
        metric_series.ewm(span=long_length, adjust=False)
        .mean()
        .rename(f"{metric_name}_ema_long")
    )
    emas_df = pd.concat([ema_short, ema_long], axis=1)
    return emas_df


def momentum(close, momentum_lookback_bars) -> pd.Series:
    """Calculate momentum series to be used as a signal.

    This indicator is later processed in decide_trades() to a weighted alpha signal.

    :param momentum_lookback_bars:
        Calculate returns based on this many bars looked back
    """
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html#pandas.DataFrame.shift
    start_close = close.shift(momentum_lookback_bars)
    momentum = (close - start_close) / start_close
    return momentum


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    indicators = IndicatorSet()

    indicators.add(
        "momentum",
        momentum,
        {"momentum_lookback_bars": parameters.momentum_lookback_bars},
        IndicatorSource.close_price,
    )

    # Social Metrics
    # Add original value indicators
    social_metrics = ["social_mentions"]

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
                "long_length": parameters.social_ema_long_length,
            },
            IndicatorSource.external_per_pair,
        )

    return indicators


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    # global max_open_positions_global  # Declare the variable as global

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
    total_equity = state.portfolio.get_total_equity()
    if total_equity > 10_000_000:
        position_valuations = "\n".join(
            [
                f"{p} (token {p.pair.base.address}): {p.get_value()}"
                for p in state.portfolio.open_positions.values()
            ]
        )
        raise RuntimeError(
            f"Portfolio total equity exceeded 1,000,000 USD. Some broken math likely happened. Total equity is {total_equity} USD.\nOpen positions:\n{position_valuations}"
        )

    #
    # Trading logic
    #
    # We do some extra checks here as we are trading low quality
    # low cap tokens which often have outright malicious data for trading.
    #

    trades = []

    # Enable trailing stop loss after we reach the profit taking level
    #
    for position in state.portfolio.open_positions.values():
        if position.trailing_stop_loss_pct is None:
            close_price = indicators.get_price(pair=position.pair)
            if (
                close_price
                and close_price
                >= position.get_opening_price()
                * parameters.trailing_stop_loss_activation_level
            ):
                position.trailing_stop_loss_pct = parameters.trailing_stop_loss_pct
        elif position.stop_loss is None:
            position.stop_loss = parameters.stop_loss_pct

    for pair in strategy_universe.iterate_pairs():

        if not is_acceptable(indicators, parameters, pair):
            # Skip this pair for the  risk management
            continue

        position_for_pair = state.portfolio.get_open_position_for_pair(pair)

        # Extract Social indicators here
        try:
            social_mentions = indicators.get_indicator_value(
                "social_mentions", pair=pair, data_delay_tolerance=pd.Timedelta(days=15)
            )
            ema_short = indicators.get_indicator_value(
                "social_mentions_emas",
                pair=pair,
                column=f"social_mentions_ema_short",
                data_delay_tolerance=pd.Timedelta(days=15),
            )
            ema_long = indicators.get_indicator_value(
                "social_mentions_emas",
                pair=pair,
                column=f"social_mentions_ema_long",
                data_delay_tolerance=pd.Timedelta(days=15),
            )
        except IndicatorDataNotFoundWithinDataTolerance:
            logger.info(
                f"Indicator data not found within tolerance for pair {pair}. Skipping this asset."
            )
            continue

        ema_short_series = indicators.get_indicator_series(
            "social_mentions_emas", pair=pair, column=f"social_mentions_ema_short"
        )
        ema_long_series = indicators.get_indicator_series(
            "social_mentions_emas", pair=pair, column=f"social_mentions_ema_long"
        )

        # Volume Based Metrics
        volume = indicators.get_price(column="volume", pair=pair)
        last_close_price = indicators.get_price(
            pair=pair
        )  # , data_lag_tolerance=pd.Timedelta(days=360))
        momentum = indicators.get_indicator_value("momentum", pair=pair)
        tvl = indicators.get_tvl(pair=pair)

        #
        # Visualisations
        #

        if input.is_visualisation_enabled():
            visualisation = (
                state.visualisation
            )  # Helper class to visualise strategy output

            visualisation.plot_indicator(
                timestamp,
                f"Social mentions {pair.base}",
                PlotKind.technical_indicator_detached,
                social_mentions,
                pair=pair,
            )
            visualisation.plot_indicator(
                timestamp,
                f"Social mentions EMA {pair.base}",
                PlotKind.technical_indicator_detached,
                ema_short,
                pair=pair,
            )
            visualisation.plot_indicator(
                timestamp,
                f"Social mentions Long {pair.base}",
                PlotKind.technical_indicator_overlay_on_detached,
                ema_long,
                pair=pair,
                detached_overlay_name=f"Social mentions EMA {pair.base}",
            )
            visualisation.plot_indicator(
                timestamp,
                f"Momentum {pair.base}",
                PlotKind.technical_indicator_detached,
                momentum,
                pair=pair,
            )

        # Check if we are too early in the backtesting to have enough data to calculate indicators
        # if None in (volume, bb_upper_interactions_social, bb_upper_interactions_social, bb_upper_sentiment_social, social_mentions, interactions, sentiment, sma):
        if None in (volume, social_mentions, tvl):  # , momentum):
            continue

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

        crossover_occurred = False
        try:
            crossover, crossover_index = contains_cross_over(
                ema_short_series,
                ema_long_series,
                lookback_period=parameters.cross_over_period,
                must_return_index=True,
            )
            crossover_occurred = crossover and (
                crossover_index >= -parameters.cross_over_period
            )
        except Exception as e:
            crossover = None
            crossover_occurred = False
            logger.trade("Cross over did not occur due to exception: %s", e)

        if (
            len(state.portfolio.open_positions) < parameters.max_assets
            and state.portfolio.get_open_position_for_pair(pair) is None
        ):
            # current_price = indicators.get_price(pair=pair)
            if momentum is None:
                if crossover_occurred and ema_short >= parameters.social_ma_min:
                    if (tvl * parameters.min_liquidity_trade_threshold) <= (
                        cash * parameters.allocation
                    ):
                        buy_amount = tvl * parameters.min_liquidity_trade_threshold
                    else:
                        buy_amount = cash * parameters.allocation

                    logger.trade(
                        "Opening position for %s with %s USDC", pair, buy_amount
                    )

                    trades += position_manager.open_spot(
                        pair,
                        value=buy_amount,
                        stop_loss_pct=parameters.stop_loss_pct,
                    )

            elif (
                crossover_occurred
                and (momentum >= parameters.minimum_mometum_threshold)
                and ema_short >= parameters.social_ma_min
            ):
                if (tvl * parameters.min_liquidity_trade_threshold) <= (
                    cash * parameters.allocation
                ):
                    buy_amount = tvl * parameters.min_liquidity_trade_threshold
                else:
                    buy_amount = cash * parameters.allocation

                logger.trade("Opening position for %s with %s USDC", pair, buy_amount)

                trades += position_manager.open_spot(
                    pair,
                    value=buy_amount,
                    stop_loss_pct=parameters.stop_loss_pct,
                )

    return trades


#
# Strategy metadata.
#
# Displayed in the user interface.
#

tags = {StrategyTag.beta, StrategyTag.live}

name = "Base Memecoin Social Trend Spotter"

short_description = "Social-data driven memecoin trading strategy on Base"

icon = ""

long_description = """
# Strategy description

This strategy leverages TrendMoon social data and market metrics to identify and trade emerging memecoin opportunities on Ethereum. By aggregating social attention around each individual token, the strategy combines data on social mentions, mindshare, and sentiment to detect new trends forming in the market. This approach works particularly well for social tokens and memecoins, as their value is heavily driven by social activity.

The strategy uses a combination of social metrics and additional market indicators, such as trading volume and momentum, to determine optimal entry points. By identifying periods of rising social interest and confirming momentum through market data, it captures upside potential while implementing risk management measures to protect against downside risks.

- Analyzes social mentions, mindshare, and sentiment to identify tokens gaining traction.
- Confirms entry signals with volume and price momentum indicators to validate trends.
- Implements robust risk management with stop-losses and trailing stops to secure profits and limit losses.
- Targets high-potential memecoins with sufficient liquidity and trading volume.
- Makes trading decisions on a daily basis, with automated execution and risk safeguards.

This strategy is designed to perform well in times of Bitcoin bullish momentum and even during sideways markets when memecoins show positive trends. It is less effective in downturns, but risk management measures help mitigate losses.

**Past performance is not indicative of future results**.

## Assets and trading venues

- Trades only on the spot market.
- Focuses on memecoin tokens on Ethereum.
- Maintains reserves in USDC stablecoin.
- Trading is executed on major DEXes (e.g., Uniswap, Sushiswap).
- Strategy decision cycle involves daily rebalancing.

## Risk

Key risk management features:

- No leverage used.
- Strict liquidity requirements.
- Multi-layer risk assessment to closely monitor:
  - Technical (price-based stops).
  - On-chain (token contract risk scoring).
  - Tax analysis (buy/sell tax limits).
- Position size limits and dynamic position sizing according to liquidity.
- Automated stop-loss monitoring on hourly timeframes.

The strategy performs best in bull markets or sideways markets with active memecoin trading. Performance may be reduced in bear markets, though risk management features help protect capital.

## Backtesting

The backtesting was performed using historical memecoin data from Base DEXes between 2024-2025.

- [See backtesting results](./backtest)
- [Read more about what is backtesting](https://tradingstrategy.ai/glossary/backtest).
- On-chain risk scoring was not included in backtesting results.

The backtesting period covered both bull and bear markets. The backtesting period saw one bull market memecoin rally from bear market lows that is unlikely to repeat in the same magnitude for the assets we trade. As such, past performance should not be used as a predictor of future returns. Backtest results have inherent limitations, and there will be variance in live performance.

## Profit

The backtested results indicate **684.39%** estimated yearly profit ([CAGR](https://tradingstrategy.ai/glossary/compound-annual-growth-rate-cagr)).

This return is based on the strategy's ability to capture upside from emerging memecoins while mitigating severe drawdowns compared to simple buy and hold.

## Benchmark

For the same backtesting period, here are some benchmark performance comparisons:

|                              | CAGR    | Maximum drawdown | Sharpe |
| ---------------------------- | ------- | ---------------- | ------ |
| This strategy                | 684.39% | -23.07%          | 2.11   |
| SP500 (20 years)             | 11%     | -33%             | 0.72   |
| Bitcoin (backtesting period) | 76%     | -76%             | 1.17   |
| Ether (backtesting period)   | 29.61%  | -45.27%          | 0.95   |

Sources:

- [Our strategy](./backtest)
- [Buy and hold BTC](./backtest)
- [Buy and hold ETH](./backtest)
- [SP500 stock index](https://curvo.eu/backtest/en/portfolio/s-p-500--NoIgygZACgBArABgSANMUBJAokgQnXAWQCUEAOAdlQEYBdeoA?config=%7B%22periodStart%22%3A%222004-02%22%7D)

## Trading frequency

The strategy is a relatively low-frequency strategy compared to other memecoin traders.

- Rebalances daily.
- Average duration of winning positions: **9 days 8 hours**.
- Average duration of losing positions: **1 day 22 hours**.
- Average positions per day: **0.33**.

## Portfolio management

- Holds a maximum of 7 concurrent positions.
- Allocates **14.2%** of the portfolio per position.
- Positions are sized based on available liquidity and overall market conditions.
- Maintains reserves in USDC as a safety net for opportunities or in case of risk events.

## Robustness

This strategy has not undergone extensive robustness testing across different market cycles, but it has been stress tested against recent data with significant market volatility, from the start of 2023.

## Updates

This strategy is one of the early social-data-driven memecoin strategies on Ethereum.

As the market evolves, it is expected that newer versions will be developed. Stay updated by [following us for updates](https://tradingstrategy.ai/community) to make sure you are trading with the latest and most optimized strategy.

## Further information

- Strategy uses **TrendMoon social data integration** for social trend detection.
  - Join our [Discord community chat](https://discord.gg/cU66eMd8) for questions or more information.
  - Check out our [Telegram channel](https://t.me/TrendMoon) for the latest discussions and insights.
  - Visit our [website](https://trendmoon.ai) to learn more about TrendMoon.
  - For all our resources, check our [link tree](https://t.co/7FJ4N8VNtB).
- Automated execution with no manual intervention required.
- Strict filters for excluding high-risk tokens.
"""
