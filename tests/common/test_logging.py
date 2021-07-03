from openfed.common.logging import logger


def test_logger():
    logger.info("INFO")
    logger.log_level("DEBUG")

    logger.log_to_file("/tmp/openfed.log")
    logger.error("ERROR")
    logger.warning("WARNING")
    logger.debug("DEBUG")

    def func():
        logger.info("FUNC", depth=2)
    func()
    # Output of LOG:
    # 2021-07-04 01:16:36.134 | ERROR    | tests.common.test_logging:test_logger:11 - ERROR
    # 2021-07-04 01:16:36.134 | WARNING  | tests.common.test_logging:test_logger:12 - WARNING
    # 2021-07-04 01:16:36.134 | DEBUG    | tests.common.test_logging:test_logger:13 - DEBUG
    # 2021-07-04 01:16:36.135 | INFO     | tests.common.test_logging:test_logger:17 - FUNC