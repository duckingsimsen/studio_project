import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(name)s :: %(levelname)s :: %(message)s"
    )
