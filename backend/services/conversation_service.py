from models.cocktail import CocktailPreferences


def _matches_removed_value(item: str, value: str) -> bool:
    item_lower = item.lower()
    value_lower = value.lower()
    return item_lower == value_lower or item_lower in value_lower or value_lower in item_lower


class ConversationService:
    def __init__(self) -> None:
        self._sessions: dict[str, CocktailPreferences] = {}
        self._last_random_cocktails: dict[str, str] = {}

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
