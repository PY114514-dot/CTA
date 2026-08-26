from datetime import date

from app.schemas import ReportCurveCandidate, ReportDisclosedMetrics, ReportLegendItem
from app.services.multi_product_report import _bind_curves_to_products


def _metric(product_id: str, name: str) -> ReportDisclosedMetrics:
    return ReportDisclosedMetrics(
        product_id=product_id,
        product_name=name,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 7, 1),
        cumulative_return=0.1,
        annualized_return=0.2,
        maximum_drawdown=-0.03,
    )


def _curve(index: int, *, color: str | None = None, label: str | None = None) -> ReportCurveCandidate:
    return ReportCurveCandidate(
        curve_index=index, left_ratio=0.1, top_ratio=0.1, right_ratio=0.9, bottom_ratio=0.4,
        color_hex=color, legend_label=label,
    )


def test_multi_product_curve_without_direct_evidence_stays_unmatched() -> None:
    bindings = _bind_curves_to_products(
        [_curve(1), _curve(2)], [_metric("a", "产品甲"), _metric("b", "产品乙")], [], [],
    )

    assert [item.product_id for item in bindings] == [None, None]
    assert all(item.status == "unmatched" for item in bindings)


def test_multi_product_binds_only_matching_legend_text() -> None:
    bindings = _bind_curves_to_products(
        [_curve(1, color="#113355")],
        [_metric("a", "产品甲"), _metric("b", "产品乙")],
        [],
        [ReportLegendItem(label="产品乙净值", color_hex="#113355", left_ratio=0.1, top_ratio=0.05, right_ratio=0.2, bottom_ratio=0.08, confidence=0.9)],
    )

    assert bindings[0].product_id == "b"
    assert bindings[0].status == "matched"


def test_single_product_still_binds_without_a_legend() -> None:
    bindings = _bind_curves_to_products([_curve(1)], [_metric("a", "产品甲")], [], [])

    assert bindings[0].product_id == "a"
    assert bindings[0].status == "matched"
