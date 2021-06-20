from datetime import timedelta

SLEEP_SHORT_TIME = timedelta(seconds=0.1).seconds  # 用于后台程序间歇
SLEEP_LONG_TIME = timedelta(seconds=5.0).seconds  # 用于后台程序间歇
SLEEP_VERY_LONG_TIME = timedelta(seconds=30.0).seconds  # 用于客户端下等待数据上
DEFAULT_PG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_LONG_TIMEOUT = timedelta(minutes=30)
DEFAULT_PG_SHORT_TIMEOUT = timedelta(seconds=1.0)

# 上次失败后，至少间隔多长时间再次尝试连接
INTERVAL_AFTER_LAST_FAILED_TIME = timedelta(seconds=10.0).seconds
# 允许的最大尝试次数，如果超过这个次数，依然没有成功，那么这个地址将会废弃
MAX_TRY_TIMES = 5
