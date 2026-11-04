# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB connector for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

This is the reference implementation of
:class:`~auto_ontology.connectors.base.SQLDatabase`.

Example
-------
::

    from duckdb_connector import DuckDB  # run from tabular-dev-tools/

    conn = DuckDB("./spider2.duckdb")
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations

import itertools
import logging
import threading
from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional

from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

_EPHEMERAL_COUNTER = itertools.count()


def _execute(
    conn: duckdb.DuckDBPyConnection, sql: str, parameters: Optional[list]
) -> pd.DataFrame:
    if parameters:
        rel = conn.execute(sql, parameters)
    else:
        rel = conn.execute(sql)
    return rel.df()


class DuckDBDatabase(SQLDatabase):
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: True).  Multiple
        processes can hold a read-only connection simultaneously; set to
        ``False`` only when you need to write to the file.
    """

    supports_statement_timeout = True

    def __init__(self, connection_string: str, *, read_only: bool = True) -> None:
        db_path = connection_string
        if db_path.startswith("duckdb://"):
            db_path = db_path[len("duckdb://") :]
        self.conn = duckdb.connect(database=db_path, read_only=read_only)
        self._connection_string: str = (
            f":memory:{next(_EPHEMERAL_COUNTER)}"
            if db_path in ("", ":memory:")
            else str(Path(db_path).resolve())
        )
        self._database_name: str = self.execute("SELECT current_database()").iloc[0, 0]
        logger.debug(
            "DuckDB connected (database=%r, read_only=%s).",
            self._database_name,
            read_only,
        )

    @property
    def dialect(self) -> str:
        """Return this engine's sqlglot dialect name.

        Must be a member of ``sqlglot.dialects.DIALECTS`` — callers pass it
        straight to sqlglot without translation. See ``CONNECTOR_REGISTRY``.
        """
        return "duckdb"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL statement and return a pandas DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters.
        timeout_s:
            Interrupt the statement after this many seconds and raise
            :class:`StatementTimeout`. DuckDB has no server-side cap, so a
            timer thread calls ``interrupt()``.
        """
        logger.debug("DuckDB executing (→ DataFrame): %s", sql[:200])
        if timeout_s is None:
            return _execute(self.conn, sql, parameters)

        # ``interrupt()`` stops whatever is running on the connection it is
        # called on, and profiling runs tables in parallel. A cursor of its own
        # (same database, separate client context) means the timer can only
        # ever stop this statement.
        cursor = self.conn.cursor()
        fired = threading.Event()

        def _interrupt() -> None:
            fired.set()
            cursor.interrupt()

        timer = threading.Timer(timeout_s, _interrupt)
        timer.daemon = True
        timer.start()
        try:
            return _execute(cursor, sql, parameters)
        except duckdb.InterruptException as exc:
            if fired.is_set():
                raise StatementTimeout(timeout_s) from exc
            raise
        finally:
            timer.cancel()
            timer.join()
            cursor.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Return all tables as a DataFrame, with their ``COMMENT ON`` text.

        Comments are not in ``information_schema``; ``duckdb_tables()`` and
        ``duckdb_views()`` carry them, so they are joined in by name.
        """
        return self.execute(
            """
            WITH comments AS (
                SELECT schema_name, table_name, comment FROM duckdb_tables()
                UNION ALL
                SELECT schema_name, view_name, comment FROM duckdb_views()
            )
            SELECT
                t.table_schema,
                t.table_name,
                c.comment AS description
            FROM information_schema.tables AS t
            LEFT JOIN comments AS c
                ON c.schema_name = t.table_schema AND c.table_name = t.table_name
            ORDER BY t.table_schema, t.table_name
        """
        )

    def get_columns(self) -> pd.DataFrame:
        """Return all columns as a DataFrame, with their ``COMMENT ON`` text."""
        return self.execute(
            """
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable = 'YES' AS is_nullable,
                c.ordinal_position,
                d.comment AS description
            FROM information_schema.columns AS c
            LEFT JOIN duckdb_columns() AS d
                ON d.schema_name = c.table_schema
                AND d.table_name = c.table_name
                AND d.column_name = c.column_name
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """
        )

    def get_queries(self) -> pd.DataFrame:
        """DuckDB has no built-in query history — loads sample queries from a CSV."""
        csv_path = (
            Path(__file__).parent
            / "benchmarks"
            / self._database_name
            / "sample_queries.csv"
        )
        if not csv_path.exists():
            logger.warning(
                "No sample queries CSV found at %s; returning empty DataFrame.",
                csv_path,
            )
            return pd.DataFrame(columns=["query_text", "end_time"])
        df = pd.read_csv(csv_path)
        df["end_time"] = datetime.today()
        return df

    def get_views(self) -> pd.DataFrame:
        """Return all views from information_schema."""
        return self.execute(
            """
            SELECT
                table_schema,
                table_name,
                view_definition
            FROM information_schema.views
            ORDER BY table_catalog, table_schema, table_name
        """
        )

    def get_pks(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fks(self) -> pd.DataFrame:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
