"""Execution state communicates the current trade execution loop state to the webhook."""
import datetime
import sys
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import dataclasses_json
from dataclasses_json import dataclass_json
from tblib import Traceback

from tradeexecutor.strategy.summary import StrategySummaryStatistics


class ExceptionData(TypedDict):
    """Serialise exception data using tblib.

    TODO: Figure out what can go here, because tblib does not provide typing.
    """
    exception_message: str
    tb_next: Optional[dict]
    tb_lineno: int


@dataclass_json
@dataclass
class LatestStateVisualisation:
    """The last visualisation of the strategy state."""

    #: When the execution state was updated last time
    last_refreshed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: 512 x 512 image PNG
    small_image: Optional[bytes] = None

    #: Dark theme version
    small_image_dark: Optional[bytes] = None

    #: 1920 x 1920 image SVG
    large_image: Optional[bytes] = None

    #: Dark theme version
    large_image_dark: Optional[bytes] = None

    def update_image_data(self,
                          small_image,
                          large_image,
                          small_image_dark,
                          large_image_dark,
                          ):
        self.small_image = small_image
        self.small_image_dark = small_image_dark
        self.large_image = large_image
        self.large_image_dark = large_image_dark
        self.last_refreshed_at = datetime.datetime.utcnow()


@dataclass_json
@dataclass
class RunState:
    """Run state.

    The status of a single trade-executor launch.

    - Anything here is not persistent, but only kept in memory
      while trade-executor is running

    A singleton instance communicates the state between
    the trade executor main loop and the webhook.

    The webhook can display the exception that caused
    the trade executor crash.

    Partially returned by different endpoints in API

    - /status

    - /source

    - /visualisation

    - /summary
    """

    #: When the execution state was updated last time
    last_refreshed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: When the executor was started
    started_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: Is the main loop alive
    #:
    #: Set to false on the crashing exception
    executor_running: bool = True

    #: The last completed trading cycle
    completed_cycle: Optional[int] = None

    #: How many cycles we have completed since launch
    cycles: int = 0

    #: How many position trigger checks we have completed since the last
    position_trigger_checks: int = 0

    #: How many position revaluations we have completed since the launch
    position_revaluations: int = 0

    #: When the executor crashed
    #:
    #: Trade execution main loop was halted by
    #: a Python exception.
    crashed_at: Optional[datetime.datetime] = None

    #: If the exception has crashed, serialise the exception information here.
    #:
    #: See :py:meth:`serialise_exception`
    exception: Optional[ExceptionData] = None

    #: The strategy source code.
    #:
    #: TODO: Move this to somewhere else long term.
    #: Use /source API endpoint to get this.
    source_code: Optional[str] = None

    #: The strategy visualisation images
    #:
    visualisation: Optional[LatestStateVisualisation] = field(default_factory=LatestStateVisualisation)

    #: Store calculated summary statistics here
    #:
    #:
    summary_statistics: StrategySummaryStatistics = field(default_factory=StrategySummaryStatistics)

    #: Measure the lag of strategy thinking and the candle data feed.
    #:
    #: What's the lag of candle creation time and when the data is processed.
    #:
    #: Regarding the candle data timestamps
    #:
    #: - The last candle should be the "real time candle" that is unfinished and started at the last minute XX:00
    #:
    #: - The candle before the last candle should be last "fully completed candle" that finished at XX:00 and no more new trades come to this candle
    #:
    #: - TimescaleDB should give us real-time data, but how much internal lag we have before a new swap hits to our
    #:   database and is rolled up the TimescaleDB hypertable is a subject to measurement,
    #:   to give an idea how close to fee of lag strategies can operate
    #:
    market_data_feed_lag: Optional[datetime.timedelta] = None

    @staticmethod
    def serialise_exception() -> ExceptionData:
        """Serialised the latest raised Python exception.

        Uses :py:mod:`tblib` to convert the Python traceback
        to something that is serialisable.
        """
        et, ev, tb = sys.exc_info()
        tb = Traceback(tb)
        data = tb.to_dict()

        # tblib loses the actual formatted exception message
        data["exception_message"] = str(ev)
        return data

    def set_fail(self):
        """Set the trade-executor main loop to a failed state.

        Reads the latest exception from Python stack and
        generates as exceptino data for it so webhook can export it.
        """
        self.exception = self.serialise_exception()
        self.last_refreshed_at = self.crashed_at = datetime.datetime.utcnow()
        self.executor_running = False

    def update_complete_cycle(self, cycle: int):
        self.completed_cycle = cycle

    def bumb_refreshed(self):
        self.last_refreshed_at = datetime.datetime.utcnow()

    def make_exportable_copy(self) -> "RunState":
        """Make a JSON serializable copy.

        Special fields like source code and images are not exported.
        """
        data = self.to_dict()
        del data["source_code"]
        del data["visualisation"]
        return RunState(**data)
