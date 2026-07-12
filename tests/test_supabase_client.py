import os
from datetime import UTC, date, datetime

import pytest

from ingestion import supabase_client
from ingestion.schema import ChainSnapshot, ContractRow


def _snapshot() -> ChainSnapshot:
    return ChainSnapshot(
        symbol="SPY", spot_price=500.0, as_of=datetime(2026, 1, 2, tzinfo=UTC),
        risk_free_rate=0.05, source="test",
        contracts=[
            ContractRow(expiry=date(2026, 1, 16), strike=500.0, right="C", bid=1.0, ask=1.2,
                        delta=0.5, gamma=0.01, theta=-0.02, vega=0.1, source="test"),
        ],
    )


def test_db_enabled_false_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert supabase_client.db_enabled() is False


def test_get_connection_none_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert supabase_client.get_connection() is None


def test_write_snapshot_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert supabase_client.write_snapshot(_snapshot()) == 0


@pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="integration test needs a real Postgres via DATABASE_URL (the shared Supabase project)",
)
def test_write_snapshot_roundtrip_against_real_db():
    conn = supabase_client.get_connection()
    try:
        n_written = supabase_client.write_snapshot(_snapshot(), conn=conn)
        assert n_written == 1

        # rerunning the same snapshot must not duplicate rows (unique constraint),
        # and the return value must reflect that -- rows actually inserted, not
        # rows submitted (fixed 2026-07-12; previously returned len(rows)
        # regardless of ON CONFLICT DO NOTHING skipping every row here).
        n_written_again = supabase_client.write_snapshot(_snapshot(), conn=conn)
        assert n_written_again == 0

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), bool_or(usable_for_trading) FROM chain_snapshots "
                "WHERE symbol = %s AND as_of = %s",
                ("SPY", datetime(2026, 1, 2, tzinfo=UTC)),
            )
            count, any_usable_for_trading = cur.fetchone()
            assert count == 1
            assert any_usable_for_trading is False  # must default False, never silently True

            # Scoped to this test's exact as_of timestamp only -- this table is a
            # shared production DB with real rows OFP's live cron has written for
            # SPY, so a blanket "WHERE symbol = 'SPY'" delete would be destructive.
            cur.execute(
                "DELETE FROM chain_snapshots WHERE symbol = %s AND as_of = %s",
                ("SPY", datetime(2026, 1, 2, tzinfo=UTC)),
            )
        conn.commit()
    finally:
        conn.close()
