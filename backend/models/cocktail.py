from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


StrengthValue = Literal["leicht", "mittel", "stark", "hoch", "alkoholfrei"]
ChatIntent = Literal[
    "conversation",
    "recommendation",
    "random",
    "catalog_query",
    "out_of_scope",
    "unknown",
]
InterpretationAction = Literal[
    "respond",
    "recommend",
    "random",
    "check_availability",
    "check_attribute",
    "list_catalog",
    "explain_recommendation",
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
    "previous_search",
    "unclear",
]
ScopeCategory = Literal["cocktail", "social", "out_of_scope"]


class ScopeAnalysis(BaseModel):
    scope: ScopeCategory
    answer: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    spelling_errors: list[str]
    eligible_word_count: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class IntentAnalysis(BaseModel):
    intent: ChatIntent
    answer: str = ""
    action: InterpretationAction = "respond"
    cocktail_name: str | None = None
    attribute: CatalogAttribute | None = None
    value: str | None = None
    context_mode: ContextMode = "unclear"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    spelling_error_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    eligible_word_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    mode: Literal["menu", "home"] = "menu"


class UnlockOrderingRequest(BaseModel):
    password: str


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


class CatalogContext(BaseModel):
    criteria: CocktailSearchCriteria | None = None
    shown_names: list[str] = Field(default_factory=list)
    referenced_cocktail: str | None = None

    model_config = ConfigDict(extra="forbid")


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
    roulette_cocktails: list[dict[str, Any]] = Field(default_factory=list)
    selected_cocktail: dict[str, Any] | None = None
    web_recipes: list[dict[str, Any]] = Field(default_factory=list)
