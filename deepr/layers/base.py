"""Interface for Layers"""

from abc import ABC
import logging
import inspect
import functools
from typing import Callable, Dict, Union, Tuple, List, Type

import tensorflow as tf

from deepr.utils.datastruct import item_to_dict, dict_to_item, to_flat_tuple


LOGGER = logging.getLogger(__name__)


class Layer(ABC):
    """Base class for composable layers in a deep learning network.

    Heavily inspired by TRAX layers, adapted for TF1.X and tf.estimator.

    Layers are the basic building block of models. A :class:`~Layer` is a
    function from one or more inputs to one or more outputs.

    The inputs of a :class:`~Layer` are tensors, packaged as follows
      - n_in = 1: one tensor (NOT wrapped in a tuple)
      - n_in > 1: a tuple of tensors

    The outputs of a :class:`~Layer` are tensors, packaged as follows
      - n_out = 1: one tensor (NOT wrapped in a tuple)
      - n_out > 1: a tuple of tensors

    The basic usage of a :class:`~Layer` is to build graphs as intuitively as
    possible. For example:

    >>> from deepr.layers import Dense
    >>> input_tensor = tf.ones([32, 8])
    >>> dense = Dense(16)
    >>> output_tensor = dense(input_tensor)
    >>> output_tensor
    <tf.Tensor 'dense/BiasAdd:0' shape=(32, 16) dtype=float32>


    Because some layers (like :class:`~Dropout`) might behave differently
    depending on the mode (TRAIN, EVAL, PREDICT), an optional argument
    can be provided:

    >>> from deepr.layers import Dropout
    >>> tensor = tf.ones([32, 8])
    >>> dropout = Dropout(0.5)
    >>> dropped = dropout(input_tensor, tf.estimator.ModeKeys.TRAIN)
    >>> not_dropped = dropout(input_tensor, tf.estimator.ModeKeys.EVAL)

    Because in a lot of cases, a :class:`~Layer` needs to be applied on a
    dictionary, yielded by a tf.data.Dataset for example, you can also
    do:

    >>> tf.reset_default_graph()
    >>> tensors = {"x": tf.ones([32, 8])}
    >>> dense = Dense(16, inputs="x", outputs="y")
    >>> tensors = dense(tensors)
    >>> tensors
    {'y': <tf.Tensor 'dense/BiasAdd:0' shape=(32, 16) dtype=float32>}

    The `inputs` and `outputs` are optional (defaults to t_0, t_1 etc.)
    and their order needs to be coherent with the order of tensors in
    tuples.

    Authors of new layer subclasses typically override one of the two
    methods of the base :class:`~Layer` class::

        def forward(self, tensors, mode: str = None):
            # tensors is either a Tensor (n_in=1) or a tuple of Tensors

        def forward_as_dict(self, tensors: Dict, mode: str = None) -> Dict:
            # tensors is a dictionary whose keys contain self.inputs

    The implementation of either of these two methods gives the
    implementation of the other for free thanks to automatic tuple to
    dictionary conversion.

    The easiest way to define custom layers is to use the :class:`~layer`
    decorator (see documentation).

    Note that layers using parameters (a :class:`~Dense` layer for example)
    should not create variables at instantiation time nor store
    variables or any other graph references as attributes.

    >>> tf.reset_default_graph()
    >>> dense = Dense(16)

    No parameters are created
    >>> dense(tf.ones([32, 8]))
    <tf.Tensor 'dense/BiasAdd:0' shape=(32, 16) dtype=float32>

    Parameters are created in the current tf.Graph

    In other words, calling the layer should not change its state. This
    is effectively enforcing functional programming. The state of the
    layer is only used to parametrize its runtime. This makes it simpler
    to define graphs with the tf.estimator API.

    If you want to define a layer and use it twice (effectively reusing
    its variables), you need to be explicit, and set the `reuse=True`
    arguments at call time. Behind the scene, it's simply wrapping the
    TF1.X variable management into a :meth:`~tf.variable_scope`.

    >>> tf.reset_default_graph()
    >>> dense = Dense(16)
    >>> dense(tf.ones([32, 8]))
    <tf.Tensor 'dense/BiasAdd:0' shape=(32, 16) dtype=float32>
    >>> dense(tf.ones([32, 8]), reuse=True)
    <tf.Tensor 'dense_1/BiasAdd:0' shape=(32, 16) dtype=float32>

    While the two operations have different names 'dense/BiasAdd:0' and
    'dense_1/BiasAdd:0', they both share the same weights.

    Good examples on how to implement parametrized layers are deepr.Dense
    and embedding.Embedding.

    Attributes
    ----------
    n_in : int
        Number of expected inputs, >= 1
    n_out : int
        Number of expected outputs, >= 1
    inputs : Union[str, Tuple[str, ...]], Optional
        Names of the n_in inputs keys in a dictionary.
        Tuple if n_in > 1, else string.
    outputs : Union[str, Tuple[str, ...]], Optional
        Names of the n_out outputs keys in a dictionary.
        Tuple if n_out > 1, else string
    name : str, optional
        Name of the layer
    """

    def __init__(
        self,
        n_in: int = None,
        n_out: int = None,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
        name: str = None,
    ):
        # Assert either number of inputs or names of inputs are given
        if n_in is None and inputs is None:
            raise ValueError("You must set either n_in or inputs (both are None)")
        if n_out is None and outputs is None:
            raise ValueError("You must set either n_out or outputs (both are None)")

        def _default_names(num: int, prefix: str = "t_"):
            if num == 1:
                return f"{prefix}0"
            else:
                return tuple(f"{prefix}{idx}" for idx in range(num))

        # Resolve n_in / inputs from arguments
        if n_in is None and inputs is not None:
            n_in = len(to_flat_tuple(inputs))
        elif n_in is not None and inputs is None:
            inputs = _default_names(n_in)

        # Resolve n_out / outputs from arguments
        if n_out is None and outputs is not None:
            n_out = len(to_flat_tuple(outputs))
        elif n_out is not None and outputs is None:
            outputs = _default_names(n_out)

        # For mypy
        assert isinstance(n_in, int)
        assert isinstance(n_out, int)

        # Store attributes
        self.n_in = n_in
        self.n_out = n_out
        self.inputs = inputs
        self.outputs = outputs
        self.name = name

        # Assert coherent attributes
        if self.n_in == 1 and not isinstance(self.inputs, str):
            msg = f"Layer {self} inputs should be a string (n_in = 1)"
            raise ValueError(msg)
        if self.n_out == 1 and not isinstance(self.outputs, str):
            msg = f"Layer {self} outputs should be a string (n_out = 1)"
            raise ValueError(msg)
        if len(to_flat_tuple(self.inputs)) != self.n_in:
            msg = f"{self}: `inputs` inconsistent with `n_in` (n_in={self.n_in}, inputs='{self.inputs})'"
            raise ValueError(msg)
        if len(to_flat_tuple(self.outputs)) != self.n_out:
            msg = f"{self}: `outputs` inconsistent with `n_out` (n_out={self.n_out}, outputs='{self.outputs}'')"
            raise ValueError(msg)

    def __repr__(self) -> str:
        args = f"{self.n_in}, {self.n_out}, inputs='{self.inputs}', outputs='{self.outputs}', name={self.name}"
        return f"{self.__class__.__name__}({args})"

    def __call__(
        self,
        tensors: Union[tf.Tensor, Dict[str, tf.Tensor], Tuple[tf.Tensor, ...]],
        mode: str = None,
        reuse: bool = False,
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor], Tuple[tf.Tensor, ...]]:
        """Forward as tuple or dictionary depending on tensors type.

        Wraps the layer call in a variable scope to be able to reuse
        variable with the ``reuse`` argument, adds a tf.identity
        operator to each output tensor using self.outputs.

        If tensors is a Dict, it returns a dictionary whose keys are
        defined by self.outputs.

        Otherwise, input tensors type is expected to be, if
            - n_in = 1: one tensor (NOT wrapped in a tuple)
            - n_in > 1: a tuple of tensors
        In that case, output tensors type is expected to be, if
            - n_out = 1: one tensor (NOT wrapped in a tuple)
            - n_out > 1: a tuple of tensors

        NOTE: Each call to this method performs inspection on the inputs
        and outputs type, which can be costly in terms of computation.

        This is not an issue when building graphs with tf.estimator as
        the graph is compiled once and for all.

        However, when using a :class:`~Layer` to preprocess a :class:`~tf.data.Dataset`
        (eg. with a ``map`` transformation), this method will be called
        for each example and might cause slowdown. It is recommended to
        explicitly use ``forward`` or ``forward_as_dict`` in that case.

        Parameters
        ----------
        tensors : Union[tf.Tensor, Dict[str, tf.Tensor], Tuple[tf.Tensor, ...]]
            Input tensors
        mode : str, optional
            One of tf.estimator.ModeKeys
        reuse : bool, optional
            Encapsulates layer call in a variable scope with reuse=reuse
        """
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            if isinstance(tensors, dict):
                # Check that tensors is coherent with self.inputs
                if not set(to_flat_tuple(self.inputs)) <= set(tensors):
                    msg = f"{self} missing inputs: {set(to_flat_tuple(self.inputs)) - set(tensors)}"
                    raise KeyError(msg)

                # Call forward_as_dict to get output tensors
                tensors_dict = self.forward_as_dict(tensors, mode)

                # Check that tensors_dict is coherent with self.outputs
                if not set(to_flat_tuple(self.outputs)) <= set(tensors_dict):
                    msg = f"{self} missing outputs: {set(to_flat_tuple(self.outputs)) - set(tensors_dict)}"
                    raise KeyError(msg)

                return tensors_dict
            else:
                # Check that tensors is coherent with self.n_in
                if self.n_in == 1 and isinstance(tensors, tuple):
                    msg = f"{self} expected 1 input, but got {tensors} (should not be a tuple)"
                    raise KeyError(msg)
                if self.n_in > 1 and len(to_flat_tuple(tensors)) != self.n_in:
                    msg = f"{self} expected {self.n_in} inputs, but got {tensors}"
                    raise KeyError(msg)

                # Call forward and convert to tuple
                tensors_tuple = self.forward(tensors, mode)

                # Check that tensors_tuple is coherent with outputs
                if len(to_flat_tuple(tensors_tuple)) != self.n_out:
                    raise IndexError(f"Expected {self.n_out} outputs but got {tensors_tuple}")

                return tensors_tuple

    def forward(
        self, tensors: Union[tf.Tensor, Tuple[tf.Tensor, ...]], mode: str = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
        """Forward method on one Tensor or a tuple of Tensors.

        Parameters
        ----------
        tensors : Union[tf.Tensor, Tuple[tf.Tensor, ...]]
            - n_in = 1: one tensor (NOT wrapped in a tuple)
            - n_in > 1: a tuple of tensors
        mode : str, optional
            Description

        Returns
        -------
        Union[tf.Tensor, Tuple[tf.Tensor, ...]]
            - n_out = 1: one tensor (NOT wrapped in a tuple)
            - n_out > 1: a tuple of tensors
        """
        try:
            return dict_to_item(self.forward_as_dict(item_to_dict(tensors, self.inputs), mode), self.outputs)
        except Exception as e:
            LOGGER.error(f"{self} error on {tensors}")
            raise e

    def forward_as_dict(self, tensors: Dict[str, tf.Tensor], mode: str = None) -> Dict[str, tf.Tensor]:
        """Forward method on a dictionary of Tensors.

        The input ``tensors`` should contain all keys defined in
        ``self.inputs`` (but might contain more keys).
        It returns a new dictionary (does not mutate the input
        ``tensors`` dictionary in-place), whose keys are exactly
        ``self.outputs``.

        Parameters
        ----------
        tensors : Dict[str, tf.Tensor]
            Dictionary mapping self.inputs to tf.Tensors.
        mode : str, optional
            One of tf.estimator.ModeKeys

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary mapping self.outputs to tf.Tensors
        """
        try:
            return item_to_dict(self.forward(dict_to_item(tensors, self.inputs), mode), self.outputs)
        except Exception as e:
            LOGGER.error(f"{self} error on {tensors}")
            raise e


def layer(
    fn: Callable = None,
    n_in: int = None,
    n_out: int = None,
    inputs: Union[str, Tuple[str, ...], List[str]] = None,
    outputs: Union[str, Tuple[str, ...], List[str]] = None,
):
    """Decorator that creates a layer constructor from a function.

    The decorator returns a subclass of :class:`~Layer`
    whose ``forward`` method is defined by the decorated function.

    For example

    >>> from deepr.layers import layer
    >>> @layer(n_in=1, n_out=1)
    ... def AddOffset(tensors, mode, offset):
    ...     return tensors + offset
    >>> add = AddOffset(offset=1)
    >>> add(1)
    2

    The class created by the decorator is roughly equivalent to

    .. code-block:: python

        class AddOffset(Layer):

            def __init__(self, offset, n_in=1, n_out=1, inputs=None, outputs=None, name=None):
                Layer.__init__(n_in=n_in, n_out=n_out, inputs=inputs, outputs=outputs, name=name)
                self.offset = offset

            def forward(self, tensors, mode: str = None):
                return tensors + self.offset

    You can also add a 'mode' argument to your layer like so
    >>> @layer(n_in=1, n_out=1)
    ... def AddOffsetInTrain(tensors, mode, offset):
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         return tensors + offset
    ...     else:
    ...         return tensors
    >>> add = AddOffsetInTrain(offset=1)
    >>> add(1, tf.estimator.ModeKeys.TRAIN)
    2
    >>> add(1, tf.estimator.ModeKeys.PREDICT)
    1

    Note that 'tensors' and 'mode' need to be the the first arguments
    of the function IN THIS ORDER.
    """
    # pylint: disable=protected-access,invalid-name
    def _create_layer_class(fn: Callable) -> Type[Layer]:
        """Decorator that creates a Layer constructor."""
        parameters = inspect.signature(fn).parameters
        signature = inspect.Signature([param for key, param in parameters.items() if key not in {"tensors", "mode"}])

        # Check parameters
        if list(parameters.keys())[0] != "tensors":
            raise TypeError(f"'tensors' should be the first parameter of {fn.__name__}")
        if "mode" in parameters:
            if list(parameters.keys())[1] != "mode":
                raise TypeError(f"'mode' should be the second parameter of {fn.__name__}")

        @functools.wraps(fn)
        def _init(self, *args, **kwargs):
            _n_in = kwargs.pop("n_in") if "n_in" in kwargs else n_in
            _n_out = kwargs.pop("n_out") if "n_out" in kwargs else n_out
            _inputs = kwargs.pop("inputs") if "inputs" in kwargs else inputs
            _outputs = kwargs.pop("outputs") if "outputs" in kwargs else outputs
            _name = kwargs.pop("name") if "name" in kwargs else None
            Layer.__init__(self, n_in=_n_in, n_out=_n_out, inputs=_inputs, outputs=_outputs, name=_name)
            signature.bind(*args, **kwargs)
            self._args = args
            self._kwargs = kwargs

        if "mode" in parameters:

            def _forward(self, tensors, mode: str = None):
                return fn(tensors, mode, *self._args, **self._kwargs)

        else:

            def _forward(self, tensors, mode: str = None):
                # pylint: disable=unused-argument
                return fn(tensors, *self._args, **self._kwargs)

        attributes = {"__module__": fn.__module__, "__doc__": fn.__doc__, "__init__": _init, "forward": _forward}
        return type(fn.__name__, (Layer,), attributes)

    if fn is not None:
        return _create_layer_class(fn)
    else:
        return _create_layer_class
