import sqlite3
from datetime import date

from app.services.futures_weekly_sqlite_import import _read_source_rows


def test_source_reader_uses_weekly_returns_when_unit_nav_resets(tmp_path):
    database = tmp_path / "futures.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE product_info (fid TEXT, name TEXT, manager TEXT, strategy_one TEXT, strategy_two TEXT, inception_date TEXT);
        CREATE TABLE weekly_return_raw (fid TEXT, week TEXT, weekly_return_value REAL);
        CREATE TABLE nav_weekly_snapshot (fid TEXT, week TEXT, unit_nav_pn REAL, cum_nav_cnw REAL, nav_cn REAL, nav_status TEXT, nav_lag_days INTEGER);
    """)
    connection.execute("INSERT INTO product_info VALUES (?, ?, ?, ?, ?, ?)", ("p1", "产品", "管理人", "主观期货", None, "2020-01-01"))
    connection.executemany(
        "INSERT INTO weekly_return_raw VALUES (?, ?, ?)",
        [("p1", date(2026, 1, 16).isoformat(), 0.0), ("p1", date(2026, 1, 23).isoformat(), 0.0019)],
    )
    connection.execute(
        "INSERT INTO nav_weekly_snapshot VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("p1", date(2026, 1, 16).isoformat(), 6.0699, 6.0699, 6.0699, "exact", 0),
    )
    connection.execute(
        "INSERT INTO nav_weekly_snapshot VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("p1", date(2026, 1, 23).isoformat(), 1.0064, 6.0813, 6.0813, "exact", 0),
    )
    connection.commit()
    connection.close()

    rows = _read_source_rows(database)

    assert len(rows) == 2
    assert rows[0]["nav"] == 1.0
    assert rows[1]["nav"] == 1.0019
