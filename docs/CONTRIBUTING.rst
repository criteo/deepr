Contributing
============

Testing and Coverage
--------------------

We aim at enforcing a coverage of at least 80% (not currently checked on
Jenkins).

Python Code Style
-----------------

Generalities
~~~~~~~~~~~~

We follow most of the PEP8 requirements with a max-line-length of 120.

Module docstring should be followed by a blank-line.

Module variables should be capitalized and separated by 2 blank lines.

Module functions and classes should be separated by 2 blank lines.

Class methods should be separated by 1 blank line

Files must end with one blank line.

.. code:: python

   """My module docstring"""

   import logging
   import os

   import numpy as np
   import mlflow

   from deepr import Model


   LOGGER = logging.getLogger(__name__)


   CONSTANT = 1


   class MyClass:
       """Docstring"""

       def __init__(self, foo):
           self.foo = foo

       def method(self):
           """Method"""


   def my_function(foo):
       """Docstring"""
       pass

Imports
~~~~~~~

Group your imports in 3 groups: ``standard`` (from python standard
library), ``external`` (external packages) and ``internal`` (current
library).

For example

.. code:: python

   import logging
   import os

   import numpy as np
   import mlflow

   from deepr_python.nn import Model

Order of Methods
~~~~~~~~~~~~~~~~

-  ``__init__``, ``__call__``, etc.
-  ``@properties``
-  ``public``
-  ``_private``
-  ``@classmethods``
-  ``@staticmethods``
-  reading order : up-to-bottom (higher level first, then helper
   functions

Docstrings
~~~~~~~~~~

We use NumPy-style docstrings.

Your docstrings must be wrapped at 72 characters (see `PEP8`_).

You are encouraged to add as much documentation as possible to classes,
including usage examples.

Example: functions
^^^^^^^^^^^^^^^^^^

.. code:: python

   def my_function(foo: str, bar: int = None) -> bool:
       """Short description on one-line

       Longer description on multiple-line. This class does this, does that
       and also etc.

       If you want to include code in your docstring, you can use
       doctest

       >>> 1 + 1
       2

       or RST code blocks

       .. code-block:: python

          result = my_function(1)

       Examples
       --------
       You can also add examples in a section like this.

       Parameters
       ----------
       foo : str
           Short description on one-line

           Longer description
       bar: int, optional
           Short description on one-line

       Returns
       -------
       bool
           Short description on one-line
       """

Example: classes
^^^^^^^^^^^^^^^^

.. code:: python

   class MyClass:
       """Short description on one-line

       Longer description on multiple-line. This class does this, does that
       and also etc.


       Examples
       --------
       You can also add examples in a section like this.

       Attributes
       ----------
       foo : str
           Short description on one-line

           Longer description
       bar: int, optional
           Short description on one-line
       """

.. _PEP8: https://www.python.org/dev/peps/pep-0008/#maximum-line-length
