from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


StrengthValue = Literal["leicht", "mittel", "stark", "hoch", "alkoholfrei"]
PreferenceStrengthValue = Literal["mild", "mittel", "stark", "hoch", "alkoholfrei"]
ChatIntent = Literal[
    "conversation",
    "recommendation",
    "random",
    "preference_update",
    "catalog_query",
    "reset_preferences",
    "out_of_scope",
    "unknown",
]
InterpretationAction = Literal[
    "respond",
    "recommend",
    "random",
    "update_preferences",
    "check_availability",
    "check_attribute",
    "list_catalog",
    "explain_recommendation",
    "reset",
    "reject",
    "clarify",
]
CatalogAttribute = Literal[
    "availability",
    "price",
    "ingredients",
    "flavor",
    "strength",
    "description",
    "recipe",
]
ContextMode = Literal[
    "new_query",
    "previous_cocktail",
    "previous_preferences",
    "unclear",
]


class IntentAnalysis(BaseModel):
    intent: ChatIntent
    answer: str = ""
    action: InterpretationAction = "respond"
    cocktail_name: str | None = None
    attribute: CatalogAttribute | None = None
    value: str | None = None
    context_mode: ContextMode = "unclear"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class PreferenceRemoveRequest(BaseModel):
    field: str
    value: str | None = None


class CocktailPreferences(BaseModel):
    liked_ingredients: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    spirits: list[str] = Field(default_factory=list)
    liked_flavors: list[str] = Field(default_factory=list)
    disliked_flavors: list[str] = Field(default_factory=list)
    strength: PreferenceStrengthValue | None = None
    alcoholic: bool | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "liked_ingredients",
        "disliked_ingredients",
        "spirits",
        "liked_flavors",
        "disliked_flavors",
        mode="before",
    )
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("preference value must be a list of strings")

        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            key = stripped.lower()
            if stripped and key not in seen:
                seen.add(key)
                normalized.append(stripped)
        return normalized

    @field_validator("strength", mode="before")
    @classmethod
    def normalize_preference_strength(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"leicht", "mild", "nicht stark", "schwach"}:
            return "mild"
        if normalized in {"strong", "kraeftig", "kräftig"}:
            return "stark"
        return normalized


class CocktailSearchCriteria(BaseModel):
    spirituose: str | None = None
    geschmack: str | None = None
    staerke: StrengthValue | None = None
    ausschluesse: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("spirituose", "geschmack", "staerke", mode="before")
    @classmethod
    def normalize_empty_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("staerke", mode="before")
    @classmethod
    def normalize_strength(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @field_validator("ausschluesse", mode="before")
    @classmethod
    def normalize_exclusions(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("ausschluesse must be a list of strings")
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class CocktailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    preis: float
    spirituose: list[str] = Field(default_factory=list)
    geschmack: list[str] = Field(default_factory=list)
    staerke: str
    zutaten: list[str] = Field(default_factory=list)
    beschreibung: str


class ChatResponse(BaseModel):
    type: Literal["recommendation", "follow_up", "message", "random"] = "recommendation"
    intent: ChatIntent | None = None
    message: str
    answer: str
    cocktails: list[dict[str, Any]] = Field(default_factory=list)
    criteria: CocktailSearchCriteria | None = None
    preferences: CocktailPreferences = Field(default_factory=CocktailPreferences)
    roulette_cocktails: list[dict[str, Any]] = Field(default_factory=list)
    selected_cocktail: dict[str, Any] | None = None
