"""Minimal working Ingestion entry point: yfinance -> clean -> Supabase.

Run: `python -m ingestion.run_yfinance_ingest [SYMBOL ...]` (default SPY).
This is the genuinely-working scraper -> Supabase write path for the
OFP-to-strategy-development Ingestion cutover -- not a mock. Every row is
written with `usable_for_trading=False` (yfinance never qualifies as
independently licensed for live capital, per the standing data-licensing
policy)."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from ingestion import clean, supabase_client
from ingestion.sources import yfinance_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_symbol(symbol: str, risk_free_rate: float) -> int:
    raw = yfinance_source.fetch_raw_chain(symbol)
    snapshot = clean.clean_raw_chain(raw, risk_free_rate)
    n_rows = supabase_client.write_snapshot(snapshot)
    logger.info("%s: fetched %d contracts, wrote %d rows -> chain_snapshots",
                symbol, snapshot.n_contracts, n_rows)
    return n_rows


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=["SPY"])
    args = parser.parse_args(argv)

    if not supabase_client.db_enabled():
        logger.error("DATABASE_URL not set -- refusing to run a no-op ingest. "
                      "Set it in .env or the environment.")
        sys.exit(1)

    risk_free_rate = yfinance_source.fetch_risk_free_rate()
    for symbol in args.symbols:
        ingest_symbol(symbol, risk_free_rate)


if __name__ == "__main__":
    main()
