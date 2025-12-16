"""
This module is wholesale lifted from the `toolz` library.

Please 'mire here: https://github.com/pytoolz/toolz
"""

from typing import Any, Callable


def pipe(data: Any, *funcs: Callable[[Any], Any]) -> Any:
    """Pipe a value through a sequence of functions

    I.e. ``pipe(data, f, g, h)`` is equivalent to ``h(g(f(data)))``

    We think of the value as progressing through a pipe of several
    transformations, much like pipes in UNIX

    ``$ cat data | f | g | h``

    >>> double = lambda i: 2 * i
    >>> pipe(3, double, str)
    '6'

    See Also:
        compose
        compose_left
        thread_first
        thread_last
    """
    for func in funcs:
        data = func(data)
    return data
