from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.routers.cta_style_risk import get_cta_style_risk
from app.services import product_store


def test_style_risk_requires_confirmed_reviewed_nav(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        product = product_store.create_product(session, standard_name="CTA")
        with pytest.raises(HTTPException, match="已确认"):
            get_cta_style_risk(product.id, session)
        product_store.confirm_product(session, product.id)
        nav = 1.0
        points = []
        for index in range(20):
            nav *= 1.001
            points.append({"observation_date": date(2024, 1, 5) + timedelta(days=index * 7), "nav": nav})
        product_store.add_nav_observations(session, product.id, points, frequency="weekly")
        observations = product_store.get_nav_series(session, product.id)
        product_store.review_nav_observations(session, [row.id for row in observations], "reviewed", "tester")
        factor_returns = np.linspace(-0.01, 0.01, 19).reshape(-1, 1)
        monkeypatch.setattr("app.routers.cta_style_risk.prepare_factor_matrix", lambda *args: {"returns": 0.001 + factor_returns[:, 0] * 0.5, "factor_returns": factor_returns, "dates": [date(2024, 1, 12) + timedelta(days=i * 7) for i in range(19)], "factor_names": ["trend"], "observation_count": 19, "warnings": []})
        body = get_cta_style_risk(product.id, session)
        assert body["data_contract"]["parsing_triggered"] is False
        assert body["method"] == "return_based_style_risk_v0"
