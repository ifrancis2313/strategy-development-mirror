"""The sole writer to the shared Supabase project's `chain_snapshots` table.

Per strategy-development's PRD ("Ingestion is the sole writer to the shared
Supabase project"), this replaces Options and Futures Pipeline's
`pipeline/block1_ingestion/db_sink.py` as the one place that writes this
table -- OFP (and every other repo) transitions to a read-only consumer.
Schema, insert/upsert logic, and the `usable_for_trading` licensing gate are
ported byte-for-byte from `db_sink.py` so a row written by this module is
indistinguishable from one OFP used to write -- `pipeline-dashboard`'s
`ofp_adapter.py` reads this exact table/shape and must keep working across
the cutover without any change on its end.

`usable_for_trading` is a hard, machine-checkable gate so a query can't
accidentally pull research-only rows into a live decision. It defaults to
(and for yfinance is always) False: per the standing data-licensing policy,
every pipeline is research/paper-trading only until backed by independently-
licensed data, and yfinance never qualifies. Flip it explicitly, per-row,
only once a source is actually confirmed licensed for live capital -- never
default it to True.

Connection: reads `DATABASE_URL` from the environment (a `.env` file via
python-dotenv if present). Use the Supabase *session pooler* host
(`aws-*.pooler.supabase.com:5432`), not the direct-connection host -- the
direct host is IPv6-only on the free tier, which breaks from IPv4-only
environments (e.g. the droplet). Same connection string OFP's own
`db_sink.py` used; see OFP's own `AGENTS.md` for the full story.
"""

import os

import psycopg2
import psycopg2.extras

from ingestion.schema import ChainSnapshot

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    spot_price DOUBLE PRECISION NOT NULL,
    risk_free_rate DOUBLE PRECISION NOT NULL,
    snapshot_source TEXT NOT NULL,
    expiry DATE NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    option_type TEXT NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    last DOUBLE PRECISION,
    volume INTEGER NOT NULL DEFAULT 0,
    open_interest INTEGER NOT NULL DEFAULT 0,
    implied_vol DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    contract_source TEXT NOT NULL,
    usable_for_trading BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (symbol, as_of, expiry, strike, option_type)
);
CREATE INDEX IF NOT EXISTS chain_snapshots_symbol_asof_idx ON chain_snapshots (symbol, as_of);
"""

_INSERT_SQL = """
INSERT INTO chain_snapshots (
    symbol, as_of, spot_price, risk_free_rate, snapshot_source,
    expiry, strike, option_type, bid, ask, last, volume, open_interest,
    implied_vol, delta, gamma, theta, vega, contract_source, usable_for_trading
) VALUES %s
ON CONFLICT (symbol, as_of, expiry, strike, option_type) DO NOTHING
RETURNING id
"""


def db_enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    return psycopg2.connect(url)


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
    conn.commit()


def write_snapshot(snapshot: ChainSnapshot, conn=None, usable_for_trading: bool = False) -> int:
    """Insert every contract row from `snapshot` into `chain_snapshots`.

    `usable_for_trading` must be left False for any source that isn't
    independently licensed for live capital (yfinance never qualifies under
    the standing data-licensing policy) -- only flip it per-call once a
    source is actually confirmed licensed.

    Returns the number of rows actually inserted -- duplicates on a rerun
    are silently skipped via the unique constraint's ON CONFLICT DO NOTHING
    and genuinely not counted, via the SQL's own RETURNING id (a row that
    hits the conflict branch never reaches RETURNING). Previously returned
    `len(rows)` (rows submitted, not rows landed) despite this same
    docstring already claiming duplicates weren't counted -- a real
    discrepancy in a live operational log line
    (`run_yfinance_ingest.py`'s "wrote %d rows"), fixed during the
    front-to-back review (2026-07-12). Returns 0 without raising if
    DATABASE_URL isn't set and no `conn` was passed in, so callers can treat
    the DB write as non-fatal.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
        if conn is None:
            return 0

    try:
        ensure_schema(conn)
        rows = [
            (
                snapshot.symbol, snapshot.as_of, snapshot.spot_price, snapshot.risk_free_rate,
                snapshot.source, c.expiry, c.strike, c.right, c.bid, c.ask, c.last, c.volume,
                c.open_interest, c.implied_vol, c.delta, c.gamma, c.theta, c.vega, c.source,
                usable_for_trading,
            )
            for c in snapshot.contracts
        ]
        with conn.cursor() as cur:
            inserted = psycopg2.extras.execute_values(cur, _INSERT_SQL, rows, fetch=True)
        conn.commit()
        return len(inserted)
    finally:
        if owns_conn:
            conn.close()
