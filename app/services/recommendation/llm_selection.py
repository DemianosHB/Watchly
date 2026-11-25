"""LLM-assisted final pick of a catalog row's titles.

The taste-profile scorer decides which candidates are worth seeing at all;
this module lets a language model make the last cut — from the head of the
scored pool it picks which titles the row finally serves, and in what order.
It is opt-in, runs only during a row build (the daily resync path, never a
catalog request), and every failure path falls back to the scored order, so a
model hiccup can never empty or shorten a row. The model can only return ids it
was given; any drift is dropped on the floor.
"""

from typing import Any

from loguru import logger

from app.core.settings import LLMConfig, resolve_llm_config
from app.models.profile import TasteProfile
from app.services.llm import llm_service
from app.services.tmdb.genre import movie_genres, series_genres

# Candidates fetched when LLM selection is on, so the model has a real choice
# beyond the row size.
POOL_FACTOR = 3

# Per-candidate blurb length in the prompt: long enough for the model to judge
# thematic fit, short enough to keep one row's prompt small.
MAX_BLURB_CHARS = 160

SELECTION_SYSTEM_PROMPT = """\
You are the final curator of a catalog row on a streaming home screen.
The recommender has already scored a pool of candidates for one viewer and one
row; your job is to pick the ones that best belong together and best match the
viewer, and order them best-first.

Judgement rules:
- Judge thematic fit and variety, not fame: no bonus for well-known titles.
- Avoid near-duplicates; a good row is coherent but varied.
- Only return ids from the candidate list, in the order to display them.
- Output a JSON array of id strings, nothing else.\
"""


def llm_selection_active(user_settings: Any, content_type: str, catalog_id: str) -> bool:
    """Whether a row build should let the model pick the final titles.

    Needs both the user's toggle and a usable LLM config; without either the
    row keeps its scored order, exactly as before.
    """
    if content_type not in ("movie", "series"):
        return False
    if not getattr(user_settings, "llm_selection", False):
        return False
    if resolve_llm_config(user_settings) is None:
        return False
    # The rewatch row serves titles from the user's own history: they are its
    # whole point, so there is nothing for a curator to pick.
    return catalog_id != "watchly.rewatch"


def taste_summary(profile: TasteProfile | None, content_type: str) -> str:
    """One line about the viewer for the prompt, or empty when there is no profile."""
    if profile is None:
        return ""
    genre_map = series_genres if content_type == "series" else movie_genres
    parts = []
    top_genres = [genre_map.get(gid, str(gid)) for gid, _ in profile.get_top_genres(limit=3)]
    if top_genres:
        parts.append("Favourite genres: " + ", ".join(top_genres))
    top_countries = [country for country, _ in profile.get_top_countries(limit=2)]
    if top_countries:
        parts.append("Preferred countries: " + ", ".join(top_countries))
    top_eras = [era for era, _ in profile.get_top_eras(limit=1)]
    if top_eras:
        parts.append(f"Preferred era: {top_eras[0]}")
    return ". ".join(parts)


def _trim_blurb(text: Any) -> str:
    """Shorten a description to the prompt budget, breaking on a word."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = " ".join(text.split())
    if len(text) <= MAX_BLURB_CHARS:
        return text
    cut = text[:MAX_BLURB_CHARS].rsplit(" ", 1)[0]
    return f"{cut}…"


def build_selection_prompt(candidates: list[dict[str, Any]], limit: int, row: str, taste: str) -> str:
    """The user prompt describing the pool to pick from."""
    lines = [
        f"Row: {row}",
        f"Viewer: {taste or 'a general audience'}",
        "",
        f"Pick the best {limit} candidates for this row, best first.",
        "",
        "Candidates (already ranked by the recommender, best first):",
    ]
    for rank, c in enumerate(candidates, start=1):
        year = c.get("releaseInfo") or ""
        year = f" ({year})" if year else ""
        genres = ", ".join(c.get("genres") or []) or "unknown genre"
        blurb = _trim_blurb(c.get("description"))
        lines.append(f"{rank}. [{c.get('id')}] {c.get('name')}{year} — {genres}" + (f" — {blurb}" if blurb else ""))
    return "\n".join(lines)


def apply_selection(pool: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    """Reorder the pool by the model's ids.

    Unknown and duplicate ids are ignored; everything the model left out keeps
    its scored order at the back, so the caller can always slice to the row
    size without coming up short.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in pool:
        by_id.setdefault(candidate.get("id"), candidate)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in ids:
        candidate = by_id.get(str(raw))
        if candidate and candidate["id"] not in seen:
            ordered.append(candidate)
            seen.add(candidate["id"])

    return ordered + [c for c in pool if c["id"] not in seen]


async def llm_select_row(
    candidates: list[dict[str, Any]],
    llm_config: LLMConfig,
    limit: int,
    row: str,
    taste: str = "",
) -> list[dict[str, Any]]:
    """Return the candidates in the model's chosen order.

    The result still holds every candidate (the caller slices to the row
    size), so a lazy or failing model degrades to a reordering at worst, and
    even that only when it returned something usable.
    """
    pool = [c for c in candidates if c.get("id") and c.get("name")]
    # Unusable candidates (no id or no name) can't be picked; keep them at the
    # end so nothing is silently dropped on either path.
    unusable = [c for c in candidates if not (c.get("id") and c.get("name"))]
    if len(pool) <= 1:
        return candidates

    prompt = build_selection_prompt(pool, limit, row, taste)
    try:
        ids = await llm_service.generate_structured(
            prompt=prompt,
            output_type=list[str],
            system_instruction=SELECTION_SYSTEM_PROMPT,
            config=llm_config,
        )
    except Exception as e:
        logger.warning(f"LLM row selection failed ({llm_config.provider}): {e}")
        return candidates

    if not isinstance(ids, list) or not ids:
        return candidates

    return apply_selection(pool, ids) + unusable
