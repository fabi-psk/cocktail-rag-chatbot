import os

from models.cocktail import CatalogContext, CocktailSearchCriteria


BAR_UNLOCK_PASSWORD = os.getenv("BAR_UNLOCK_PASSWORD", "1111")


class ConversationService:
    def __init__(self) -> None:
        self._last_random_cocktails: dict[str, str] = {}
        self._catalog_contexts: dict[str, CatalogContext] = {}
        self._ordering_blocks: set[str] = set()

    def reset_session(self, session_id: str | None) -> None:
        if session_id:
            self._last_random_cocktails.pop(session_id, None)
            self._catalog_contexts.pop(session_id, None)

    def block_ordering(self, session_id: str | None) -> None:
        if session_id:
            self._ordering_blocks.add(session_id)

    def is_ordering_blocked(self, session_id: str | None) -> bool:
        return bool(session_id and session_id in self._ordering_blocks)

    def unlock_ordering(self, session_id: str | None, password: str) -> bool:
        if not session_id or password != BAR_UNLOCK_PASSWORD:
            return False
        self._ordering_blocks.discard(session_id)
        return True

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

conversation_service = ConversationService()
