from app.services.vlm_evaluation import evaluate_manifest, evaluate_structure_prediction


def test_evaluator_scores_structure_without_calling_a_model() -> None:
    expected = {"recognizable": True, "plot_bbox_1000": [100, 100, 900, 900], "curves": [{"name": "产品"}], "y_ticks": ["0.8", "1.0"]}
    predicted = {"plot_bbox_1000": [110, 110, 890, 890], "curves": [{"name": "产品"}], "y_ticks": ["0.8", "1.0"]}
    result = evaluate_structure_prediction(expected, predicted)
    assert result["status"] == "passed"
    assert result["metrics"]["plot_bbox_iou"] > 0.9


def test_evaluator_requires_rejection_for_unreadable_chart() -> None:
    report = evaluate_manifest([{"expected": {"recognizable": False}, "predicted": {"rejected": True}}])
    assert report["passed_count"] == 1
