# pylint: disable=unused-import,missing-docstring

from deepr.utils.broadcasting import make_same_shape
from deepr.utils.checkpoint import save_variables_in_ckpt
from deepr.utils.datastruct import to_flat_tuple, item_to_dict, dict_to_item
from deepr.utils.exceptions import handle_exceptions
from deepr.utils.field import Field, TensorType
from deepr.utils.graph import (
    get_feedable_tensors,
    get_fetchable_tensors,
    import_graph_def,
    get_by_name,
    INIT_ALL_TABLES,
)
from deepr.utils.iter import progress, chunks
from deepr.utils.uuid import msb_lsb_to_str, str_to_msb_lsb
from deepr.utils.tables import TableContext, table_from_file, table_from_mapping, index_to_string_table_from_file
import deepr.utils.mlflow
import deepr.utils.graphite
