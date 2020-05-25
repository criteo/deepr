DeepR: Build and Train Deep Learning Pipelines for Production
=============================================================

DeepR is a library for Deep Learning on top of Tensorflow 1.x that focuses on production capabilities. It makes it easy to define pipelines (via the ``Job`` abstraction), preprocess data (via the ``Prepro`` abstraction), design models (via the ``Layer`` abstraction) and train them either locally or on a Yarn cluster. It also integrates nicely with MLFlow and Graphite, allowing for production ready logging capabilities.

It can be seen as a collection of generic tools and abstractions to be extended for more specific use cases. See the ``Use DeepR`` section for more information.

Submitting jobs and defining flexible pipelines is made possible thanks to a config system based off simple dictionaries and import strings. It is similar to `Thinc config system <https://thinc.ai/docs>`_ or `gin config <https://github.com/google/gin-config>`_ in a lot of ways.


Why a Deep Learning Library based on TF1.x
------------------------------------------

Tensorflow 1.x provides great production oriented capabilities, centered around the ``tf.Estimator`` API. It makes it possible to deploy models using a ``protobuf`` with no ``python`` code, and optimize computational graphs with XLA compilation.

Although ``DeepR`` comes with a ``Layer`` interface (most similar to `google TRAX <https://github.com/google/trax>`_ and very close to most modern frameworks) that makes it easy to define models using a functional programming approach, most of its capabilities are orthogonal to it. Most of the building blocks expect generic ``python`` types (for example, a ``Layer`` is merely a function ``fn(tensors, mode)``).


Use DeepR
---------

You can use ``DeepR`` as a simple python library, reusing only a subset of the concepts (the config system is generic for example) or build your own extension as a standalone python package that depends on ``deepr``.

Have a look at the submodule `example <../deepr/example>`_ of ``deepr`` that illustrates what a package built on top of deepr would look like. It defines custom jobs, layers, preprocessors, macros as well as `configs <../deepr/example/configs>`_. Once your custom components are packaged in a library, it is easy to run configs with


.. code-block::

    deepr run config.json macros.json



Installation
------------

Prerequisites
~~~~~~~~~~~~~

Make sure you use ``python>=3.6`` and an up-to-date version of ``pip`` and ``setuptools``

.. code-block::

    python --version
    pip install -U pip setuptools

It is recommended to install ``deepr`` in a new virtual environment. For example

.. code-block::

    python -m venv deepr
    source deepr/bin/activate
    pip install -U pip setuptools
    pip install deepr[cpu]


Using Pip
~~~~~~~~~

If installing using pip and your own ``requirements.txt`` file, be aware that ``Tensorflow`` is listed in ``extras_require`` in the ``setup.py``, which means that ``pip install deepr`` WON'T INSTALL Tensorflow. This is because the Tensorflow requirement is different depending on the platform (GPU or CPU-only).

You can specify which extras to use using the ``[cpu]`` or ``[gpu]`` argument like in the following examples

.. code-block::

    pip install deepr[cpu]
    pip install deepr[gpu]
    pip install -e ".[cpu]"
    pip install -e ".[gpu]"

Or alternatively, pre-install Tensorflow separately like so

.. code-block::

    pip install tensorflow==1.15.2
    pip install deepr



From Source
~~~~~~~~~~~

First, clone the ``deepr`` repo on your local machine with

.. code-block::

    git clone https://github.com/criteo/deepr.git
    cd deepr

To install from source in editable mode, run

.. code-block::

    make install-cpu

Or to install on a GPU enabled machine

.. code-block::

    make install-gpu

To install development tools and test requirements, run


.. code-block::

    make install-dev

Test
----

To run unit tests in your current environment, run

.. code-block::

    make test


To run unit tests + lint in a fresh virtual environment, run


.. code-block::

    make venv-lint-test


Lint
----

To run ``mypy``, ``pylint`` and ``black --check``:

.. code-block::

    make lint

To auto-format the code using ``black``

.. code-block::

    make black


Command Line Tools
------------------

To get a list of available commands, run

.. code-block::

    deepr --help

Contributing
------------

See `CONTRIBUTING <CONTRIBUTING.rst>`_


Change log
----------

See `CHANGELOG <CHANGELOG.rst>`_
