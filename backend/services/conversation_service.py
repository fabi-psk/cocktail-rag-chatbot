from models.cocktail import CatalogContext, CocktailPreferences, CocktailSearchCriteria


def _matches_removed_value(item: str, value: str) -> bool:
    item_lower = item.lower()
    value_lower = value.lower()
    return item_lower == value_lower or item_lower in value_lower or value_lower in item_lower


class ConversationService:
    def __init__(self) -> None:
        self._sessions: dict[str, CocktailPreferences] = {}
        self._last_random_cocktails: dict[str, str] = {}
        self._catalog_contexts: dict[str, CatalogContext] = {}

    def get_preferences(self, session_id: str | None) -> CocktailPreferences:
        if not session_id:
            return CocktailPreferences()
        return self._sessions.get(session_id, CocktailPreferences())

    def update_preferences(self, session_id: str | None, preferences: CocktailPreferences) -> None:
        if session_id:
            self._sessions[session_id] = preferences

    def reset_preferences(self, session_id: str | None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)
            self._last_random_cocktails.pop(session_id, None)
            self._catalog_contexts.pop(session_id, None)

    def get_catalog_context(self, session_id: str | None) -> CatalogContext:
        if not session_id:
            return CatalogContext()
        return self._catalog_contexts.get(session_id, CatalogContext())

    def record_catalog_search(
        self,
        session_id: str | None,
        criteria: CocktailSearchCriteria,
        shown_names: list[str],
        append: bool = False,
    ) -> None:
        if not session_id:
            return
        current = self.get_catalog_context(session_id)
        previous_names = current.shown_names if append and current.criteria == criteria else []
        names = list(dict.fromkeys(previous_names + shown_names))
        self._catalog_contexts[session_id] = CatalogContext(
            criteria=criteria,
            shown_names=names,
            referenced_cocktail=None,
        )

    def set_referenced_cocktail(
        self,
        session_id: str | None,
        cocktail_name: str | None,
    ) -> None:
        if not session_id or not cocktail_name:
            return
        current = self.get_catalog_context(session_id)
        self._catalog_contexts[session_id] = current.model_copy(
            update={"referenced_cocktail": cocktail_name}
        )

    def get_last_random_cocktail(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        return self._last_random_cocktails.get(session_id)

    def update_last_random_cocktail(self, session_id: str | None, cocktail_name: str | None) -> None:
        if session_id and cocktail_name:
            self._last_random_cocktails[session_id] = cocktail_name

    def remove_preference(self, session_id: str | None, field: str, value: str | None = None) -> CocktailPreferences:
        preferences = self.get_preferences(session_id)
        if not hasattr(preferences, field):
            return preferences

        current_value = getattr(preferences, field)
        if isinstance(current_value, list):
            if value is None:
                updated_value = []
            else:
                updated_value = [item for item in current_value if not _matches_removed_value(item, value)]
            preferences = preferences.model_copy(update={field: updated_value})
        elif field in {"strength", "alcoholic"}:
            preferences = preferences.model_copy(update={field: None})

        if value and field in {"disliked_ingredients", "disliked_flavors"}:
            preferences = preferences.model_copy(update={
                "disliked_ingredients": [
                    item for item in preferences.disliked_ingredients if not _matches_removed_value(item, value)
                ],
                "disliked_flavors": [
                    item for item in preferences.disliked_flavors if not _matches_removed_value(item, value)
                ],
            })

        self.update_preferences(session_id, preferences)
        return preferences


conversation_service = ConversationService()
