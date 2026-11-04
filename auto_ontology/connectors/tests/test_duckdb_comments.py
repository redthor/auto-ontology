# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB ``COMMENT ON`` text reaches the catalog as ``description``."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from auto_ontology.connectors.duckdb import DuckDBDatabase


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE trades (id INTEGER, px DOUBLE)")
    conn.execute("CREATE TABLE plain (a INTEGER)")
    conn.execute("CREATE VIEW v AS SELECT id FROM trades")
    conn.execute("COMMENT ON TABLE trades IS 'Executed trades'")
    conn.execute("COMMENT ON COLUMN trades.px IS 'Execution price'")
    conn.execute("COMMENT ON VIEW v IS 'Trade ids'")
    conn.close()
    database = DuckDBDatabase(str(path))
    yield database
    database.close()


def test_columns_carry_comments(db):
    cols = db.get_columns().set_index(["table_name", "column_name"])["description"]
    assert cols[("trades", "px")] == "Execution price"
    assert pd.isna(cols[("trades", "id")])
    assert pd.isna(cols[("plain", "a")])


def test_tables_and_views_carry_comments(db):
    tables = db.get_tables().set_index("table_name")["description"]
    assert tables["trades"] == "Executed trades"
    assert tables["v"] == "Trade ids"
    assert pd.isna(tables["plain"])


def test_opens_read_only_by_default(db):
    with pytest.raises(duckdb.Error):
        db.execute("CREATE TABLE nope (x INTEGER)")
