# pylint: disable=unused-import,missing-docstring

from deepr.layers.base import Layer, layer
from deepr.layers.bpr import BPR, MaskedBPR
from deepr.layers.bpr_max import BPRMax, MaskedBPRMax
from deepr.layers.click_rank import ClickRank
from deepr.layers.combinators import Sequential, Select, Rename, Parallel, ActiveMode, Scope
from deepr.layers.core import (
    Add,
    Concat,
    Conv1d,
    Dense,
    DotProduct,
    ExpandDims,
    Identity,
    LogicalAnd,
    LogicalOr,
    Product,
    Scale,
    Softmax,
    Sum,
    ToFloat,
)
from deepr.layers.dropout import SpatialDropout1D, Dropout
from deepr.layers.embedding import Embedding, CombineEmbeddings
from deepr.layers.lookup import (
    Lookup,
    LookupFromFile,
    LookupFromMapping,
    LookupIndexToString,
    index_to_string_table_from_file,
    table_from_file,
    table_from_mapping,
)
from deepr.layers.mask import Equal, NotEqual, BooleanMask, BooleanReduceMode
from deepr.layers.nce_loss import NegativeSampling, MaskedNegativeSampling
from deepr.layers.reduce import Average, WeightedAverage
from deepr.layers.size import IsMinSize
from deepr.layers.slice import Slice, SliceFirst, SliceLast, SliceLastPadded
from deepr.layers.sparse import ToDense
from deepr.layers.string import StringJoin
from deepr.layers.top_one import TopOne, MaskedTopOne
from deepr.layers.top_one_max import TopOneMax, MaskedTopOneMax
from deepr.layers.transformer import (
    Transformer,
    FeedForward,
    Normalization,
    PositionalEncoding,
    SelfMultiheadAttention,
    AttentionMask,
)
from deepr.layers.triplet_precision import TripletPrecision
