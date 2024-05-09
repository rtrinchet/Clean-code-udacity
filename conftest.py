import logging

def pytest_configure():
    logfile = "logs/test_log.log"
    logging.basicConfig(filename=logfile,
                        level=logging.INFO,
                        filemode="w",
                        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
