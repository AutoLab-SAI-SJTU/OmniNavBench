"""Built-in dataset adapters.

Importing this package triggers the registration of all built-in adapters
via the ``@register_adapter`` decorator.  :mod:`bench.datasets.loader`
imports this package automatically, so end users generally do not need to
import it directly.

To add a new adapter:

1. Create ``bench/datasets/adapters/my_adapter.py`` with a class decorated
   by ``@register_adapter``.
2. Add an import below so it is executed when this package is loaded.
"""

from . import native  # noqa: F401 – registers NativeAdapter
from . import sage3d  # noqa: F401 – registers Sage3DAdapter
from . import vlnce  # noqa: F401 – registers VlnceAdapter

__all__ = ["native", "sage3d", "vlnce"]
