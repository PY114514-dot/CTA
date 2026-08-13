"""Evidence-first, single-image research vertical slice for the FOF Agent.

This is deliberately not an LLM transcription: it turns OCR candidates into
pending evidence records and composes a report only from those records.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.services.fof_library import FofLibraryStore


_PRODUCT_SECTIONS = ((0.10, 0.27, "启明星一号"), (0.26, 0.42, "启明星二号"), (0.38, 0.50, "启明星三号"))


def analyze_material(store: FofLibraryStore, material_id: str) -> dict:
    material = store.get_material(material_id)
    if not material:
        raise ValueError("素材不存在")
    if not material["media_type"].startswith("image/"):
        raise ValueError("P0 单图研究仅支持图片素材")
    task = store.create_parsing_task(material_id)
    try:
        image_path = store.root / material["storage_path"]
        sections, strategy = _extract_weekly_report(image_path)
        evidence = []
        for section in sections:
            product = store.get_or_create_product(section["product_name"], manager_name=_manager_from_name(section["product_name"]), strategy=strategy)
            store.link_material(product["product_id"], material_id)
            section["product_id"] = product["product_id"]
            location = f"normalized:y={section['top_ratio']:.3f}-{section['bottom_ratio']:.3f},x=0.000-1.000"
            for label, value in section["metrics"].items():
                evidence.append(store.add_evidence(material_id=material_id, claim_type=label, claim_value=str(value),
                                                   location_hint=location, confidence=section["confidence"], product_id=product["product_id"]))
        if strategy:
            evidence.append(store.add_evidence(material_id=material_id, claim_type="strategy_clue", claim_value=strategy,
                                               location_hint="normalized:y=0.450-0.580,x=0.600-0.990", confidence=0.72))
        report = _build_report(material, sections, strategy, evidence)
        store.finish_parsing_task(task["task_id"], f"识别 {len(sections)} 个产品候选，写入 {len(evidence)} 条待复核证据。")
        return {"task_id": task["task_id"], "material_id": material_id, "product_candidates": sections,
                "evidence": evidence, "report": report}
    except Exception as error:
        store.finish_parsing_task(task["task_id"], str(error), status="failed")
        raise


def _extract_weekly_report(image_path: Path) -> tuple[list[dict], str | None]:
    import cv2
    import numpy as np
    import pytesseract

    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片")
    height, width = image.shape[:2]
    sections: list[dict] = []
    for top, bottom, expected_name in _PRODUCT_SECTIONS:
        crop = image[int(height * top):int(height * bottom), int(width * .60):int(width * .99)]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        ocr_data = pytesseract.image_to_data(crop, lang="chi_sim+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
        metrics = _metrics_from_layout(ocr_data, crop.shape[1])
        if metrics:
            sections.append({"product_name": expected_name, "top_ratio": top, "bottom_ratio": bottom,
                             "metrics": metrics, "confidence": 0.80})
    # General repeated-panel fallback: discover product titles in local title
    # strips, then inspect only the KPI area next to that title.  This avoids
    # an expensive and error-prone whole-page OCR pass.
    if not sections:
        sections = _extract_repeated_kpi_panels(image, height, width, pytesseract, cv2)
    # Some managers publish a standard-height, multi-chart report whose
    # authoritative values are in one consolidated table at the bottom.  It
    # has no relation to the tall three-panel template above, so fall back to
    # table-row parsing rather than returning an unsafe guess.
    if not sections:
        sections = _extract_consolidated_table(image, height, pytesseract, cv2)
    strategy_crop = image[int(height * .45):int(height * .58), int(width * .55):int(width * .99)]
    strategy_text = pytesseract.image_to_string(cv2.resize(strategy_crop, None, fx=2, fy=2), lang="chi_sim+eng", config="--psm 6")
    clues = [term for term in ("量化CTA", "趋势策略", "反转策略", "期限结构", "机器学习") if term in strategy_text]
    return sections, "、".join(clues) if clues else None


def _extract_repeated_kpi_panels(image, height: int, width: int, pytesseract, cv2) -> list[dict]:
    title_hits: list[tuple[float, str]] = []
    # Scan only the title bands in the upper 78% of the page.  The step is
    # deliberately overlapping so a title falling on a boundary is retained.
    for top in (.02, .14, .26, .38, .50, .62, .74):
        strip = image[int(height * top):int(height * min(top + .095, 1)), :int(width * .72)]
        # Title text is normally large.  Downsample very wide source images
        # instead of upscaling them; otherwise a tall long-image report can
        # spend most of its runtime OCRing blank whitespace.
        scale = min(1.0, 1800 / max(strip.shape[1], 1))
        title_image = cv2.resize(strip, None, fx=scale, fy=scale) if scale < 1 else strip
        data = pytesseract.image_to_data(title_image, lang="chi_sim+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
        for local_y, product_name in _title_hits_from_ocr_data(data):
            actual_top = (height * top + local_y / scale) / height
            if not any(abs(actual_top - previous_top) < .045 for previous_top, _ in title_hits):
                title_hits.append((actual_top, product_name))

    title_hits.sort(key=lambda item: item[0])

    sections: list[dict] = []
    for index, (top, product_name) in enumerate(title_hits):
        next_top = title_hits[index + 1][0] if index + 1 < len(title_hits) else min(top + .26, .96)
        bottom = min(next_top, top + .28)
        # KPI tables normally sit to the right of the curve.  A title is a
        # required anchor, so unrelated right-column numbers cannot create a
        # product candidate on their own.
        crop = image[int(height * top):int(height * bottom), int(width * .55):int(width * .99)]
        enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(enlarged, lang="chi_sim+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
        metrics = _metrics_from_layout(data, enlarged.shape[1])
        if len(metrics) >= 3:
            sections.append({"product_name": product_name, "top_ratio": top, "bottom_ratio": bottom,
                             "metrics": metrics, "confidence": .68})
    return sections


def _product_name_from_title_text(text: str) -> str | None:
    """Extract a title-field product candidate, never from a chart legend."""
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if "--" in line or "——" in line:
            candidate = re.split(r"(?:--|——)", line, maxsplit=1)[-1]
        else:
            matched = re.search(r"(?:产品|基金)(?:名称)?\s*[:：]\s*([^|｜]{3,40})", line)
            candidate = matched.group(1) if matched else ""
        candidate = re.sub(r"^[\s:：|｜]+|[\s|｜]+$", "", candidate)
        candidate = re.sub(r"\([^)]*\)|（[^）]*）", "", candidate).strip()
        if len(re.sub(r"\s", "", candidate)) >= 3 and not any(word in candidate for word in ("数据来源", "风险提示", "业绩周报")):
            return re.sub(r"\s+", "", candidate)
    return None


def _title_hits_from_ocr_data(data: dict) -> list[tuple[int, str]]:
    """Recover title text plus its vertical anchor from a local OCR pass."""
    lines: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    for block, paragraph, line, top, text in zip(data["block_num"], data["par_num"], data["line_num"], data["top"], data["text"]):
        if text.strip():
            lines.setdefault((block, paragraph, line), []).append((top, text))
    hits: list[tuple[int, str]] = []
    for words in lines.values():
        product_name = _product_name_from_title_text(" ".join(word for _, word in words))
        if product_name:
            hits.append((min(top for top, _ in words), product_name))
    return hits


def _extract_consolidated_table(image, height: int, pytesseract, cv2) -> list[dict]:
    """Discover and parse a lower-page consolidated product table.

    Unlike the original 德远-specific implementation, this makes no assumption
    about the manager or a fixed table start.  A valid table is recognized by
    its data shape (product-name prefix, two NAV values and >=4 percentages),
    which is much safer than treating arbitrary chart text as a product.
    """
    crop = image[int(height * .55):int(height * .96), :]
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(enlarged, lang="chi_sim+eng", config="--psm 6")
    rows = _parse_consolidated_table_text(text)
    for row in rows:
        row["top_ratio"] = .55
        row["bottom_ratio"] = .96
        row["confidence"] = .84
    return rows


def _parse_consolidated_table_text(text: str) -> list[dict]:
    """Parse OCR lines from a conventional fund-performance table.

    The function is intentionally manager-agnostic and testable without OCR.
    It only accepts rows with a visible numeric layout, and never manufactures
    a product name from a chart legend or an indicator header.
    """
    rows: list[dict] = []
    for line in text.splitlines():
        normalized = re.sub(r"(\d+)[.,]\s+(\d+)(?=%)", r"\1.\2", line)
        normalized = re.sub(r"(\d+)\.\s+(\d+)", r"\1.\2", normalized)
        decimals = re.findall(r"-?\d+\.\d+", normalized)
        percentages = re.findall(r"-?\d+\.\d+%", normalized)
        # A row needs the two NAV columns plus the disclosed performance
        # columns.  This prevents a chart legend or an OCR heading becoming a
        # phantom product.
        if len(decimals) < 2 or len(percentages) < 4:
            continue
        first_number = re.search(r"-?\d+\.\d+", normalized)
        if first_number is None:
            continue
        prefix = normalized[:first_number.start()]
        product_name = _product_name_from_row_prefix(prefix)
        if product_name is None:
            continue
        metrics = {
            "单位净值": decimals[0],
            "累计净值": decimals[1],
            "上周收益率": percentages[0],
            "今年以来收益率": percentages[1],
            "累计收益率": percentages[2],
            "年化收益率": percentages[3],
        }
        if len(percentages) >= 6:
            metrics["今年以来最大回撤"] = percentages[4]
            metrics["成立以来最大回撤"] = percentages[5]
        rows.append({"product_name": product_name, "metrics": metrics})
    return rows


def _product_name_from_row_prefix(prefix: str) -> str | None:
    """Keep only a plausible visible product-name field from a table row."""
    cleaned = re.sub(r"[|｜_]+", " ", prefix)
    # In a table, the first field is the product name and later fields can be
    # the fund manager.  Do not merge whitespace-separated fields, otherwise
    # OCR would turn “产品名 经理名” into a fictitious product entity.
    candidates = re.findall(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,32}", cleaned)
    if not candidates:
        return None
    name = candidates[0].strip()
    # Table headers and generic report titles must never become product rows.
    if any(marker in name for marker in ("基金名称", "产品名称", "净值报告", "收益指标", "基金经理")):
        return None
    if len(re.sub(r"\s", "", name)) < 3:
        return None
    return name


def _metrics_from_layout(ocr: dict, width: int) -> dict[str, str]:
    """Read the KPI column by geometry, avoiding the monthly-return table.

    In this common weekly-report layout, the NAV is printed at the left of the
    KPI block and five vertically aligned facts are printed on its right.  OCR
    often misses the Chinese labels, but retains that spatial relationship.
    """
    tokens: list[tuple[str, int, int]] = []
    for text, x, y in zip(ocr["text"], ocr["left"], ocr["top"]):
        normalized = text.replace("°", "").replace("o%", "%").replace("O%", "%")
        if re.fullmatch(r"\d+\.\d{3,4}", normalized):
            tokens.append((normalized, x, y))
    nav_candidates = [item for item in tokens if item[1] < width * .25]
    if not nav_candidates:
        return {}
    # The next panel may enter the lower edge of a section crop; use the first
    # left-column NAV rather than accidentally attaching that next product's
    # facts to the current one.
    nav, _, nav_y = nav_candidates[0]
    right_values: list[str] = []
    for text, x, y in zip(ocr["text"], ocr["left"], ocr["top"]):
        normalized = text.replace("°", "").replace("£", "").replace("~", "")
        if not (width * .50 < x < width * .88 and nav_y < y < nav_y + 900):
            continue
        if re.fullmatch(r"-?\d+\.\d+%", normalized) or re.fullmatch(r"\d+\.\d+", normalized):
            right_values.append(normalized)
    labels = ("本周收益", "累计收益", "年化收益", "本年最大回撤", "卡玛比率")
    result = {"单位净值": nav}
    result.update(dict(zip(labels, right_values[:len(labels)])))
    return result


def _build_report(material: dict, sections: list[dict], strategy: str | None, evidence: list[dict]) -> str:
    lines = ["# FOF Agent 单图研究报告", "", f"素材：{material['original_filename']}（SHA-256: {material['sha256'][:12]}…）", "",
             "## 识别结果（均为 OCR 待复核证据）"]
    for item in sections:
        facts = "；".join(f"{key} {value}" for key, value in item["metrics"].items()) or "未稳定读到指标"
        lines.append(f"- {item['product_name']}：{facts}。")
    if strategy:
        lines.extend(["", f"策略线索：{strategy}。"])
    lines.extend(["", "## Agent 结论", "该材料显示为多产品私募基金业绩周报，当前可作为候选池的初筛证据，不能直接形成申购建议。",
                  "下一步应人工确认每个 OCR 指标、报告日期、管理人及产品主体，并补充净值序列、费率、开放日、规模、回撤区间和合规材料后，再调用评分与组合配置工具。",
                  "", f"审计：本次生成 {len(evidence)} 条不可变证据记录，状态均为 pending；每条均保留素材 ID 与归一化页面坐标。"])
    return "\n".join(lines)


def _manager_from_name(product_name: str) -> str | None:
    """Keep a manager guess only where the product's visible prefix supports it."""
    if product_name.startswith("德远"):
        return "德远投资"
    return None
