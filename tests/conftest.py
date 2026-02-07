import os


def pytest_configure():
    os.environ.setdefault("MIH_DISABLE_TRANSFORMERS", "1")
