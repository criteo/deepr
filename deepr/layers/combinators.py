"""Combinators layers"""

from typing import Dict, List, Union, Tuple, Generator

from deepr.layers.base import Layer
from deepr.utils.datastruct import to_flat_tuple


class Sequential(Layer):
    """Class to easily compose layers in a deep learning network.

    A Deep Learning Network is a Directed Acyclic Graph (DAG) of layers.
    The easiest way to define a DAG is by stacking layers on top of each
    others. For example:

    .. code-block:: python

        @dprl.layer(n_in=1, n_out=1)
        def OffsetLayer(tensors, mode, offset):
            return tensors + offset

        layer = dprl.Sequential(
            OffsetLayer(offset=1, inputs="x"),
            OffsetLayer(offset=2, outputs="y")
        )
        layer(1)  # (1 + 1) + 2 = 4
        layer({"x": 1})  # {"y": 4}

    Because in some cases your model is more complicated (branches etc.)
    you can exploit the inputs / outputs naming capability of the base
    :class:`~Layer` class. For example:

    .. code-block:: python

        @dprl.layer(n_in=2, n_out=1)
        def Add(tensors, mode):
            x, y = tensors
            return x + y

        layer = dprl.Sequential(
            OffsetLayer(offset=2, inputs="x", outputs="y"),
            OffsetLayer(offset=2, inputs="x", outputs="z"),
            Add(inputs="y, z", outputs="total"),
        )
        layer(1)  # (1 + 2) + (1 + 2) = 6
        layer({"x": 1})  # {"total": 6}

    As always, the resulting layer can be operated on Tensors or
    dictionaries of Tensors. The inputs / outputs of the :class:`~Sequential`
    layer corresponds to the inputs of the first layer and the outputs
    of the last layer in the stack (intermediary nodes that are not
    returned by the last layer will not be returned).

    An easy way to define arbitrary inputs / outputs nodes is to use the
    :class:`~Select` class. For example:

    .. code-block:: python

        layer = dprl.Sequential(
            dprl.Select("x1, x2"),
            OffsetLayer(offset=2, inputs="x1", outputs="y1"),
            OffsetLayer(offset=2, inputs="x2", outputs="y2"),
            Add(inputs="y1, y2", outputs="y3"),
            dprl.Select("y1, y2, y3"),
        )
        layer((1, 2))  # (3, 4, 7)
        layer({"x1": 1, "x2": 2})  # {"y1": 3, "y2": 4, "y3": 7}

    Note that default naming still applies, so it won't raise an error
    if you try stacking layers with incoherent shapes, as long as the
    correctly named nodes are defined.

    .. code-block:: python

        layer = dprl.Sequential(
            dprl.Select(n_in=2),  # Defines "t_0" and "t_1" nodes
            OffsetLayer(offset=2),  # Replace "t_0" <- "t_0" + 2
            Add(),  # Returns "t_0" + "t_1"
        )
        result = layer((tf.constant(2), tf.constant(2)))
        with tf.Session() as sess:
            assert sess.run(result) == 6
    """

    def __init__(self, *layers: Union[Layer, List[Layer], Tuple[Layer], Generator[Layer, None, None]]):
        self.layers = to_flat_tuple(layers)
        super().__init__(
            n_in=len(to_flat_tuple(self.layers[0].inputs)),
            n_out=len(to_flat_tuple(self.layers[-1].outputs)),
            inputs=self.layers[0].inputs,
            outputs=self.layers[-1].outputs,
        )
        # Check consistency of inputs / outputs of intermediate layers
        keys = set(to_flat_tuple(self.inputs))
        for layer in self.layers:
            for key in to_flat_tuple(layer.inputs):
                if key not in keys:
                    raise ValueError(f"Input '{key}' of layer {layer} not found")
            keys.update(to_flat_tuple(layer.outputs))

    def forward_as_dict(self, tensors: Dict, mode: str = None) -> Dict:
        """Forward method of the layer"""
        new_tensors = dict()  # type: Dict
        for layer in self.layers:
            outputs = layer.forward_as_dict({**tensors, **new_tensors}, mode)
            new_tensors.update(outputs)
        return outputs


class Select(Layer):
    """Layer to extract inputs / outputs from previous layers

    The :class:`~Select` layer is particularly useful when defining
    arbitrary DAGs of layers : it is a convenient way to select which
    nodes should be inputs, and which should be outputs. For example:

    .. code-block:: python

        layer = dprl.Select(inputs=("x", "y"), outputs="z", n_in=2, indices=1)
        layer((1, 2))  # 2
        layer({"x": 1, "y": 2})  # {"z": 2}

    See :class:`~Sequential` documentation for more precisions.
    """

    def __init__(
        self,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
        indices: List[int] = None,
        n_in: int = None,
    ):
        if n_in is None and inputs is None:
            msg = "`n_in` and `inputs` cannot both be `None`"
            raise ValueError(msg)
        if n_in is None:
            n_in = len(to_flat_tuple(inputs))
        self.indices = to_flat_tuple(indices) if indices is not None else list(range(n_in))
        if inputs is not None and outputs is None:
            outputs = tuple(inputs[idx] for idx in self.indices)
        super().__init__(n_in=n_in, n_out=len(self.indices), inputs=inputs, outputs=outputs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensors = to_flat_tuple(tensors)
        result = tuple(tensors[idx] for idx in self.indices)
        if len(result) == 1:
            return result[0]
        else:
            return result


class ActiveMode(Layer):
    """Active Mode Layer."""

    def __init__(
        self,
        layer: Layer,
        mode: Union[str, Tuple[str, ...]] = None,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
    ):
        if inputs is None:
            inputs = layer.inputs
        if outputs is None:
            outputs = layer.outputs
        super().__init__(n_in=layer.n_in, n_out=layer.n_out, inputs=inputs, outputs=outputs, name=layer.name)
        self.layer = layer
        self.mode = to_flat_tuple(mode)
        if self.n_in != self.n_out:
            raise ValueError("Number of inputs / outputs must be the same/")

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        if self.mode is not None and mode is not None and mode not in self.mode:
            return tensors
        return self.layer.forward(tensors, mode)


class Rename(Layer):
    """Wrap Layer in a Node to rename inputs / outputs.

    Allows you to rename inputs / outputs nodes of a :class:`~Layer`
    instance. This can be useful if you end up with a :class:`~Layer`
    instance with inputs and outputs name that are not suitable for your
    needs.

    For example:

    .. code-block:: python

        @dprl.layer(n_in=2, n_out=1)
        def Add(tensors):
            x, y = tensors
            return x + y

        add = Add(inputs="a, b", outputs="c")
        layer = dprl.Rename(layer=add, inputs="x, y", outputs="z")
        layer((1, 1))  # 2
        layer({"x": 1, "y": 1})  # {"z": 2}

    Note that the same behavior can be achieved using :class:`~Select`
    and :class:`~Sequential` as follows:

    .. code-block:: python

        layer = dprl.Sequential(
            dprl.Select(inputs=("x", "y"), outputs=("a", "b")),
            Add(inputs=("a", "b"), outputs="c"),
            dprl.Select("c", "z"),
        )
    """

    def __init__(
        self,
        layer: Layer,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
    ):
        if inputs is None:
            inputs = layer.inputs
        if outputs is None:
            outputs = layer.outputs
        super().__init__(n_in=layer.n_in, n_out=layer.n_out, inputs=inputs, outputs=outputs, name=layer.name)
        self.layer = layer

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return self.layer.forward(tensors, mode)


class Parallel(Layer):
    """Apply layers in parallel on consecutive inputs.

    If you have 2 layers F(a, b) -> x and G(c) -> (y, z), it defines a
    layer H(a, b, c) -> (x, y, z). For example:

    .. code-block:: python

        layer1 = Add(inputs="x1, x2", outputs="y1")
        layer2 = OffsetLayer(offset=1, inputs="x3", outputs="y2")
        layer = dprl.Parallel(layer1, layer2)
        layer((1, 1, 2))  # (2, 3)
        layer({"x1": 1, "x2": 1, "x3": 2})  # {"y1": 2, "y2": 3}
    """

    def __init__(self, *layers: Union[Layer, List[Layer], Tuple[Layer], Generator[Layer, None, None]]):
        self.layers = to_flat_tuple(layers)
        n_in = sum(layer.n_in for layer in self.layers)
        n_out = sum(layer.n_out for layer in self.layers)
        inputs = to_flat_tuple([layer.inputs for layer in self.layers])
        outputs = to_flat_tuple([layer.outputs for layer in self.layers])
        inputs = (inputs if n_in > 1 else inputs[0]) if len(set(inputs)) == n_in else None
        outputs = (outputs if n_out > 1 else outputs[0]) if len(set(outputs)) == n_out else None
        super().__init__(n_in=n_in, n_out=n_out, inputs=inputs, outputs=outputs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensors = to_flat_tuple(tensors)
        new_tensors, idx = [], 0
        for layer in self.layers:
            tensors_in = tensors[idx] if layer.n_in == 1 else tuple(tensors[idx : idx + layer.n_in])
            new_tensors.append(layer.forward(tensors_in, mode=mode))
            idx += layer.n_in
        result = to_flat_tuple(new_tensors)
        return result if self.n_out > 1 else result[0]
