import asyncio

import app.services.recommendation.llm_selection as llm_selection
from app.core.settings import LLMConfig, UserSettings, get_default_settings
from app.models.profile import TasteProfile
from app.services.recommendation.llm_selection import (
    MAX_BLURB_CHARS,
    apply_selection,
    build_selection_prompt,
    llm_select_row,
    llm_selection_active,
    taste_summary,
)

CONFIG = LLMConfig(provider="openrouter", api_key="test-key")


def candidate(i: int, blurb: str = "Blurb", **extra) -> dict:
    return {
        "id": f"tt{i}",
        "type": "movie",
        "name": f"Title {i}",
        "releaseInfo": "2026",
        "genres": ["Drama"],
        "description": blurb,
        "_tmdb_id": i,
        **extra,
    }


def run(coro):
    return asyncio.run(coro)


def test_apply_selection_reorders_and_keeps_everything():
    pool = [candidate(1), candidate(2), candidate(3)]

    assert apply_selection(pool, ["tt3", "tt1", "tt2"]) == [candidate(3), candidate(1), candidate(2)]


def test_apply_selection_ignores_unknown_and_duplicate_ids():
    pool = [candidate(1), candidate(2), candidate(3)]

    reordered = apply_selection(pool, ["tt9", "tt2", "tt2", "tt7", "tt1"])

    assert reordered == [candidate(2), candidate(1), candidate(3)]


def test_apply_selection_fills_unpicked_from_scored_order():
    pool = [candidate(i) for i in range(1, 6)]

    reordered = apply_selection(pool, ["tt4"])

    assert reordered == [candidate(4), candidate(1), candidate(2), candidate(3), candidate(5)]


def test_llm_select_row_uses_the_model_order(monkeypatch):
    async def fake_generate(prompt, output_type, system_instruction, config):
        assert "tt" in prompt
        assert output_type == list[str]
        return ["tt3", "tt1"]

    monkeypatch.setattr(llm_selection.llm_service, "generate_structured", fake_generate)
    pool = [candidate(1, blurb="x" * 400), candidate(2), candidate(3)]

    results = run(llm_select_row(pool, CONFIG, limit=2, row="watchly.rec"))

    assert [m["id"] for m in results] == ["tt3", "tt1", "tt2"]
    assert results[0] is pool[2] and results[1] is pool[0]


def test_llm_select_row_fails_open(monkeypatch):
    pool = [candidate(1), candidate(2), candidate(3)]

    async def exploding(prompt, output_type, system_instruction, config):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_selection.llm_service, "generate_structured", exploding)
    assert run(llm_select_row(pool, CONFIG, limit=2, row="watchly.rec")) == pool

    monkeypatch.setattr(llm_selection.llm_service, "generate_structured", lambda *a, **k: None)
    assert run(llm_select_row(pool, CONFIG, limit=2, row="watchly.rec")) == pool

    async def empty(prompt, output_type, system_instruction, config):
        return []

    monkeypatch.setattr(llm_selection.llm_service, "generate_structured", empty)
    assert run(llm_select_row(pool, CONFIG, limit=2, row="watchly.rec")) == pool


def test_llm_select_row_skips_the_call_for_tiny_pools(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("no LLM call expected")

    monkeypatch.setattr(llm_selection.llm_service, "generate_structured", unexpected)
    nosy = [{"id": "kd1", "type": "movie", "name": "No id here"}]

    # Too small to be worth a call, and unusable entries ride along unchanged.
    assert run(llm_select_row([candidate(1), nosy[0]], CONFIG, limit=2, row="watchly.rec")) == [candidate(1), nosy[0]]


def test_build_selection_prompt_stays_within_blurb_budget():
    long_winded = {"id": "tt1", "name": "Windy", "releaseInfo": "2026"}
    long_winded["genres"] = ["Drama"]
    long_winded["description"] = "word " * 200

    prompt = build_selection_prompt(
        [candidate(2), long_winded], limit=20, row="watchly.rec", taste="Favourite genres: Drama"
    )

    assert "[tt2] Title 2 (2026) — Drama — Blurb" in prompt
    assert "Row: watchly.rec" in prompt
    assert "Viewer: Favourite genres: Drama" in prompt
    assert "Pick the best 20 candidates" in prompt
    assert len(long_winded["description"]) > MAX_BLURB_CHARS
    wordy_line = next(line for line in prompt.splitlines() if "Windy" in line)
    assert "…" in wordy_line and len(wordy_line) < MAX_BLURB_CHARS + 80


def test_taste_summary_reads_the_profile():
    profile = TasteProfile(genre_scores={18: 9.0, 27: 4.0}, country_scores={"KR": 3.0}, era_scores={"2010s": 2.0})

    summary = taste_summary(profile, "movie")

    assert "Drama" in summary
    assert "KR" in summary
    assert "2010s" in summary
    assert taste_summary(None, "movie") == ""


def test_llm_selection_active_needs_toggle_key_and_a_normal_catalog():
    on = UserSettings(
        catalogs=[], llm=LLMConfig(provider="openrouter", api_key="k"), llm_selection=True
    )
    no_key = UserSettings(catalogs=[], llm_selection=True)
    default = get_default_settings()

    assert llm_selection_active(on, "movie", "watchly.rec")
    assert llm_selection_active(on, "series", "watchly.theme.2")
    assert not llm_selection_active(no_key, "movie", "watchly.rec")
    assert not llm_selection_active(default, "movie", "watchly.rec")
    assert not llm_selection_active(on, "all", "watchly.rec")
    assert not llm_selection_active(on, "movie", "watchly.rewatch")


def test_old_accounts_without_the_flag_default_off():
    assert not llm_selection_active(UserSettings(catalogs=[], llm=CONFIG), "movie", "watchly.rec")
