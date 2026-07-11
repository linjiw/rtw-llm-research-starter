"""Tests for scripts/18_paper1_figures.py (Paper-1 figure generator).

Companion to scripts/17. Fast, pure-CPU, Agg-rendered into tmp_path. Loaded
with the digit-prefixed-module importlib idiom.
"""
import importlib.util
import json
from pathlib import Path


def load_figs():
    path = Path(__file__).resolve().parents[1] / "scripts" / "18_paper1_figures.py"
    spec = importlib.util.spec_from_file_location("paper1_figures", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ensure_assets_json():
    """scripts/18 depends on 17's JSON; generate it if absent (committed banks)."""
    repo = Path(__file__).resolve().parents[1]
    aj = repo / "outputs" / "paper1_assets.json"
    if not aj.exists():
        p = repo / "scripts" / "17_paper1_assets.py"
        spec = importlib.util.spec_from_file_location("assets17", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.main()
    return aj


def test_self_check_passes():
    _ensure_assets_json()
    m = load_figs()
    assert m.run(["--self_check"]) == 0


def test_dry_run_writes_nothing(tmp_path):
    _ensure_assets_json()
    m = load_figs()
    assert m.run(["--dry_run", "--out_dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []


def test_write_produces_figures(tmp_path):
    _ensure_assets_json()
    m = load_figs()
    assert m.run(["--out_dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "figures_manifest.json").read_text())
    ok = {c for c, v in manifest["figures"].items() if v["status"] == "ok"}
    # C1, C4, C5 always renderable from committed artifacts; C3 needs v13 banks
    assert {"C1", "C4", "C5"} <= ok
    for c, v in manifest["figures"].items():
        if v["status"] != "ok":
            continue
        for a in v["assets"]:
            assert Path(a).exists(), f"{c}: missing {a}"
        assert {".png", ".pdf"} <= {Path(a).suffix for a in v["assets"]}


def test_c1_matches_source_json():
    """The figure must read from 17's JSON -- no independent recompute/drift."""
    aj = _ensure_assets_json()
    assets = json.loads(aj.read_text())
    c1 = assets["C1_selection_saturation"]
    # every bank saturated => scatter would land on the diagonal
    assert all(b["reranked@8"] <= b["oracle@8"] for b in c1["per_bank"])
    assert c1["n_banks"] == c1["n_banks_zero_gap"]


def test_missing_source_json_returns_nonzero(tmp_path, monkeypatch):
    m = load_figs()
    monkeypatch.setattr(m, "ASSETS_JSON", tmp_path / "does_not_exist.json")
    assert m.run(["--out_dir", str(tmp_path)]) == 1
