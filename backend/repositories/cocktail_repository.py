import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger("cocktail-rag")
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "cocktails.json"


class CocktailRepositoryError(RuntimeError):
    pass


class CocktailRepository:
    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or DEFAULT_DATA_PATH

    def list_all(self) -> list[dict[str, Any]]:
        try:
            with self.data_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except OSError as exc:
            logger.exception("cocktails.json is not readable: %s", exc)
            raise CocktailRepositoryError("Cocktail-Daten konnten nicht gelesen werden.") from exc
        except json.JSONDecodeError as exc:
            logger.exception("cocktails.json is invalid JSON: %s", exc)
            raise CocktailRepositoryError("Cocktail-Daten sind kein gueltiges JSON.") from exc

        if not isinstance(data, list):
            raise CocktailRepositoryError("Cocktail-Daten muessen eine Liste sein.")
        return data

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        normalized_name = name.lower().strip()
        for cocktail in self.list_all():
            if cocktail.get("name", "").lower().strip() == normalized_name:
                return cocktail
        return None
