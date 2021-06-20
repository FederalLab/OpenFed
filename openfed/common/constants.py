from datetime import timedelta

# thread
SLEEP_SHORT_TIME = 0.1
SLEEP_LONG_TIME = 5.0
SLEEP_VERY_LONG_TIME = 30.0

# communication
DEFAULT_PG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_LONG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_SHORT_TIMEOUT = timedelta(seconds=1.0)
INTERVAL_AFTER_LAST_FAILED_TIME = 10.0
# if exceed this time, the address will be discarded.
MAX_TRY_TIMES = 5
