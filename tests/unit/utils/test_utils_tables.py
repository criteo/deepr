"""Test for utils.tables"""

import pytest

import deepr


def test_tables_context():
    """Test TableContext"""
    with deepr.utils.TableContext() as tables:
        # Active context
        assert deepr.utils.TableContext.active() is not None

        # Missing table
        with pytest.raises(KeyError):
            tables.get("my_table")

        # Creation of new table
        table = deepr.utils.table_from_mapping(name="my_table", mapping={1: 2})
        assert table is not None

        # Reuse via context
        assert tables.get("my_table") is table

        # Reuse via table function
        reused = deepr.utils.table_from_mapping(name="my_table", reuse=True)
        assert reused is table

        # Duplicate
        with pytest.raises(ValueError):
            deepr.utils.table_from_mapping(name="my_table", mapping={1: 2})
