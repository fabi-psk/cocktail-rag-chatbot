import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from models.cocktail import CocktailSearchCriteria
from repositories.cocktail_repository import CocktailRepository


UMLAUT_REPLACEMENTS = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
})

TYPO_ALIASES = {
    "alkholfrei": "alkoholfrei",
    "alkolfrei": "alkoholfrei",
    "alkoholfre": "alkoholfrei",
    "fruchtg": "fruchtig",
    "fruchtich": "fruchtig",
    "kremig": "cremig",
    "suer": "sauer",
    "sauerr": "sauer",
    "suess": "suess",
    "suss": "suess",
    "wodka": "vodka",
    "vodcka": "vodka",
    "votka": "vodka",
    "tequilla": "tequila",
    "teqila": "tequila",
    "kokuss": "kokos",
    "cocos": "kokos",
    "rhum": "rum",
}

STRENGTH_ALIASES = {
    "stark": {"hoch", "stark"},
    "hoch": {"hoch", "stark"},
    "mittel": {"mittel"},
    "leicht": {"leicht"},
    "alkoholfrei": {"alkoholfrei"},
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.lower().translate(UMLAUT_REPLACEMENTS)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return TYPO_ALIASES.get(normalized, normalized)


def fuzzy_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def fuzzy_token_match(term: str, text: str) -> bool:
    if len(term) < 4:
        return False

    tokens = text.split()
    for token in tokens:
        if len(token) >= 4 and fuzzy_similarity(term, token) >= 0.82:
            return True

    for idx in range(len(tokens) - 1):
        combined = f"{tokens[idx]} {tokens[idx + 1]}"
        if fuzzy_similarity(term, combined) >= 0.86:
            return True

    return False


def term_matches(term: str | None, text: str | None) -> bool:
    term_lower = normalize_text(term)
    text_lower = normalize_text(text)
    if not term_lower or not text_lower:
        return False

    if len(term_lower) <= 3:
        return bool(re.search(rf"\b{re.escape(term_lower)}\b", text_lower))

    if term_lower in text_lower or (len(text_lower) >= 4 and text_lower in term_lower):
        return True

    if fuzzy_token_match(term_lower, text_lower):
        return True

    for suffix in ["en", "es", "er", "e", "n", "s"]:
        if term_lower.endswith(suffix):
            stem = term_lower[:-len(suffix)]
            if len(stem) > 3 and (
                stem in text_lower
                or text_lower in stem
                or fuzzy_similarity(stem, text_lower) >= 0.85
            ):
                return True
        for token in text_lower.split():
            if token.endswith(suffix):
                stem = token[:-len(suffix)]
                if len(stem) > 3 and (
                    stem in term_lower
                    or term_lower in stem
                    or fuzzy_similarity(stem, term_lower) >= 0.85
                ):
                    return True

    return False


def field_contains(term: str | None, values: list[str] | str | None) -> bool:
    if values is None:
        return False
    if isinstance(values, str):
        values = [values]
    return any(term_matches(term, value) for value in values)


def cocktail_contains(cocktail: dict[str, Any], term: str) -> bool:
    searchable_values: list[str] = [
        cocktail.get("name", ""),
        cocktail.get("beschreibung", ""),
        cocktail.get("staerke", ""),
    ]
    searchable_values.extend(cocktail.get("spirituose", []))
    searchable_values.extend(cocktail.get("geschmack", []))
    searchable_values.extend(cocktail.get("zutaten", []))
    return any(term_matches(term, value) for value in searchable_values)


def matches_strength(criteria_strength: str | None, cocktail: dict[str, Any]) -> bool:
    if not criteria_strength:
        return True
    accepted = STRENGTH_ALIASES.get(normalize_text(criteria_strength), {normalize_text(criteria_strength)})
    cocktail_strength = normalize_text(cocktail.get("staerke", ""))
    return cocktail_strength in accepted


def score_cocktail(cocktail: dict[str, Any], criteria: CocktailSearchCriteria) -> int:
    score = 0
    if criteria.spirituose and field_contains(criteria.spirituose, cocktail.get("spirituose", [])):
        score += 30
    if criteria.geschmack and field_contains(criteria.geschmack, cocktail.get("geschmack", [])):
        score += 20
    if criteria.staerke and matches_strength(criteria.staerke, cocktail):
        score += 15
    return score


def search_cocktails(
    criteria: CocktailSearchCriteria,
    cocktails: list[dict[str, Any]] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    cocktails = cocktails if cocktails is not None else CocktailRepository().list_all()

    scored: list[tuple[int, dict[str, Any]]] = []
    has_positive_criteria = any([
        criteria.spirituose,
        criteria.geschmack,
        criteria.staerke,
    ])

    for cocktail in cocktails:
        if any(cocktail_contains(cocktail, exclusion) for exclusion in criteria.ausschluesse):
            continue
        if criteria.spirituose and not field_contains(criteria.spirituose, cocktail.get("spirituose", [])):
            continue
        if criteria.geschmack and not field_contains(criteria.geschmack, cocktail.get("geschmack", [])):
            continue
        if not matches_strength(criteria.staerke, cocktail):
            continue

        score = score_cocktail(cocktail, criteria)
        if not has_positive_criteria:
            score = 1
        if score > 0:
            cocktail_copy = cocktail.copy()
            cocktail_copy["match_score"] = score
            scored.append((score, cocktail_copy))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [cocktail for _, cocktail in scored[:limit]]


def search_by_query_params(
    spirituose: str | None = None,
    geschmack: str | None = None,
    staerke: str | None = None,
) -> list[dict[str, Any]]:
    criteria = CocktailSearchCriteria(
        spirituose=spirituose,
        geschmack=geschmack,
        staerke=staerke,
    )
    return search_cocktails(criteria, limit=1000)
