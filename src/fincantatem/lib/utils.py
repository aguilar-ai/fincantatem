"""
This module is wholesale lifted from the `toolz` library.

Please 'mire here: https://github.com/pytoolz/toolz

---

Copyright (c) 2013 Matthew Rocklin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of toolz nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import operator
from functools import partial, reduce
from operator import attrgetter
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


def get_in(keys, coll, default=None, no_default=False):
    """Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
    ``no_default`` is specified, then it raises KeyError or IndexError.

    ``get_in`` is a generalization of ``operator.getitem`` for nested data
    structures such as dictionaries and lists.

    >>> transaction = {'name': 'Alice',
    ...                'purchase': {'items': ['Apple', 'Orange'],
    ...                             'costs': [0.50, 1.25]},
    ...                'credit card': '5555-1234-1234-1234'}
    >>> get_in(['purchase', 'items', 0], transaction)
    'Apple'
    >>> get_in(['name'], transaction)
    'Alice'
    >>> get_in(['purchase', 'total'], transaction)
    >>> get_in(['purchase', 'items', 'apple'], transaction)
    >>> get_in(['purchase', 'items', 10], transaction)
    >>> get_in(['purchase', 'total'], transaction, 0)
    0
    >>> get_in(['y'], {}, no_default=True)
    Traceback (most recent call last):
        ...
    KeyError: 'y'

    See Also:
        itertoolz.get
        operator.getitem
    """
    try:
        return reduce(operator.getitem, keys, coll)
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default


def instanceproperty(fget=None, fset=None, fdel=None, doc=None, classval=None):
    """Like @property, but returns ``classval`` when used as a class attribute

    >>> class MyClass(object):
    ...     '''The class docstring'''
    ...     @instanceproperty(classval=__doc__)
    ...     def __doc__(self):
    ...         return 'An object docstring'
    ...     @instanceproperty
    ...     def val(self):
    ...         return 42
    ...
    >>> MyClass.__doc__
    'The class docstring'
    >>> MyClass.val is None
    True
    >>> obj = MyClass()
    >>> obj.__doc__
    'An object docstring'
    >>> obj.val
    42
    """
    if fget is None:
        return partial(
            instanceproperty, fset=fset, fdel=fdel, doc=doc, classval=classval
        )
    return InstanceProperty(fget=fget, fset=fset, fdel=fdel, doc=doc, classval=classval)


class InstanceProperty(property):
    """Like @property, but returns ``classval`` when used as a class attribute

    Should not be used directly.  Use ``instanceproperty`` instead.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, classval=None):
        self.classval = classval
        property.__init__(self, fget=fget, fset=fset, fdel=fdel, doc=doc)

    def __get__(self, obj, type=None):
        if obj is None:
            return self.classval
        return property.__get__(self, obj, type)

    def __reduce__(self):
        state = (self.fget, self.fset, self.fdel, self.__doc__, self.classval)
        return InstanceProperty, state


def return_none(exc):
    """Returns None."""
    return None


class excepts:
    """A wrapper around a function to catch exceptions and
    dispatch to a handler.

    This is like a functional try/except block, in the same way that
    ifexprs are functional if/else blocks.

    Examples
    --------
    >>> excepting = excepts(
    ...     ValueError,
    ...     lambda a: [1, 2].index(a),
    ...     lambda _: -1,
    ... )
    >>> excepting(1)
    0
    >>> excepting(3)
    -1

    Multiple exceptions and default except clause.

    >>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
    >>> excepting([])
    >>> excepting([1])
    1
    >>> excepting({})
    >>> excepting({0: 1})
    1
    """

    def __init__(self, exc, func, handler=return_none):
        self.exc = exc
        self.func = func
        self.handler = handler

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except self.exc as e:
            return self.handler(e)

    @instanceproperty(classval=__doc__)
    def __doc__(self):
        from textwrap import dedent

        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = "(%s)" % ", ".join(
                    map(attrgetter("__name__"), exc),
                )
            else:
                exc_name = exc.__name__

            return dedent(
                """\
                A wrapper around {inst.func.__name__!r} that will except:
                {exc}
                and handle any exceptions with {inst.handler.__name__!r}.

                Docs for {inst.func.__name__!r}:
                {inst.func.__doc__}

                Docs for {inst.handler.__name__!r}:
                {inst.handler.__doc__}
                """
            ).format(
                inst=self,
                exc=exc_name,
            )
        except AttributeError:
            return type(self).__doc__

    @property
    def __name__(self):
        exc = self.exc
        try:
            if isinstance(exc, tuple):
                exc_name = "_or_".join(map(attrgetter("__name__"), exc))
            else:
                exc_name = exc.__name__
            return f"{self.func.__name__}_excepting_{exc_name}"
        except AttributeError:
            return "excepting"
