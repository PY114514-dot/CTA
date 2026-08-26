"""PG 方言分支的编译级测试（批次 13 精度）。

不要求本机有 PostgreSQL：SQLAlchemy 的方言编译可以在纯 Python 下验证
SQLite/PG 两个分支各自生成正确的 SQL。覆盖两处生产方言分支：
1. product_archive._universe_size_expr：SQLite json_extract vs PG JSONB 下标；
2. fof_library.FofLibraryStore._insert_ignore：INSERT OR IGNORE vs ON CONFLICT。
"""

from types import SimpleNamespace

from sqlalchemy.dialects import postgresql, sqlite

from app.routers.product_archive import _universe_size_expr
from app.services.fof_library import FofLibraryStore


class _FakeSession:
    """只暴露 _universe_size_expr 需要的 get_bind().dialect.name。"""

    def __init__(self, dialect_name: str) -> None:
        self._dialect = SimpleNamespace(name=dialect_name)

    def get_bind(self):
        return SimpleNamespace(dialect=self._dialect)


class _FakeConnection:
    """捕获 _insert_ignore 拼出的 SQL 文本，而不是真正执行。"""

    def __init__(self, dialect_name: str) -> None:
        self.dialect = SimpleNamespace(name=dialect_name)
        self.captured: str | None = None

    def execute(self, statement, values) -> None:  # noqa: ANN001 - 仅测试桩
        self.captured = str(statement)


def test_universe_size_expr_sqlite_uses_json_extract() -> None:
    expression = _universe_size_expr(_FakeSession("sqlite"))
    # literal_binds 把 JSON 路径字面量内联进 SQL，否则路径以 ? 绑定形式出现。
    sql = str(expression.compile(
        dialect=sqlite.dialect(),
        compile_kwargs={"literal_binds": True},
    ))
    assert "json_extract" in sql
    assert "universe_size" in sql


def test_universe_size_expr_postgres_uses_jsonb_subscript() -> None:
    expression = _universe_size_expr(_FakeSession("postgresql"))
    sql = str(expression.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    ))
    # PG 走 JSONB 下标路径，不允许再出现 SQLite 专有的 json_extract。
    assert "json_extract" not in sql
    assert "universe_size" in sql


def test_insert_ignore_sqlite_uses_insert_or_ignore() -> None:
    connection = _FakeConnection("sqlite")
    FofLibraryStore._insert_ignore(connection, "products", "name", {"name": "测试产品"})
    assert connection.captured is not None
    assert "INSERT OR IGNORE" in connection.captured
    assert "ON CONFLICT" not in connection.captured


def test_insert_ignore_postgres_uses_on_conflict() -> None:
    connection = _FakeConnection("postgresql")
    FofLibraryStore._insert_ignore(connection, "products", "name", {"name": "测试产品"})
    assert connection.captured is not None
    assert "ON CONFLICT DO NOTHING" in connection.captured
    assert "INSERT OR IGNORE" not in connection.captured


def test_insert_ignore_binds_named_parameters() -> None:
    connection = _FakeConnection("sqlite")
    FofLibraryStore._insert_ignore(
        connection, "products", "name, manager_name",
        {"name": "测试产品", "manager_name": "测试管理人"},
    )
    assert connection.captured is not None
    assert ":name" in connection.captured
    assert ":manager_name" in connection.captured
