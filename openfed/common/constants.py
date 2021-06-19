from datetime import timedelta

SLEEP_SHORT_TIME = timedelta(seconds=0.1).seconds  # 用于后台程序间歇
SLEEP_LONG_TIME = timedelta(seconds=5.0).seconds  # 用于后台程序间歇
SLEEP_VERY_LONG_TIME = timedelta(seconds=30.0).seconds  # 用于客户端下等待数据上
DEFAULT_PG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_LONG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_SHORT_TIMEOUT = timedelta(seconds=1.0)