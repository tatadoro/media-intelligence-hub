# sitecustomize.py
# Автоматически импортируется Python при старте (если лежит в sys.path).
# Нужен для совместимости pymorphy2 с Python 3.11+ (inspect.getargspec удалён).

import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec  # type: ignore[attr-defined]