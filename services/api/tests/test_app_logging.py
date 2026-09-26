import logging

import pytest

import app.main as main_module


@pytest.fixture
def app_logger() -> logging.Logger:
    """The ``app`` logger, restored to its prior handlers and level after the test."""
    logger = logging.getLogger("app")
    handlers = list(logger.handlers)
    level = logger.level
    try:
        yield logger
    finally:
        logger.handlers = handlers
        logger.setLevel(level)


def test_configure_logging_installs_exactly_one_handler(app_logger: logging.Logger) -> None:
    main_module.configure_logging()
    main_module.configure_logging()

    installed = [h for h in app_logger.handlers if isinstance(h, main_module._AppLogHandler)]
    assert len(installed) == 1
    assert app_logger.level == logging.INFO


def test_configure_logging_prints_level_and_logger_name(
    app_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    main_module.configure_logging()

    logging.getLogger("app.loop").info("hello")

    assert capsys.readouterr().err == "INFO app.loop: hello\n"
