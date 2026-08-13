import asyncio
import unittest
from unittest.mock import patch

from models.cocktail import CocktailSearchCriteria
from services import chat_service
from services.conversation_service import ConversationService


COCKTAILS = [
    {
        "name": "Caribbean Dream",
        "kategorie": "Cremig",
        "preis": 8.5,
        "spirituose": ["Rum"],
        "geschmack": ["cremig", "fruchtig"],
        "staerke": "mittel",
        "zutaten": ["Rum", "Ananassaft", "Sahne"],
        "beschreibung": "Cremig und fruchtig.",
    },
    {
        "name": "Bahama Mama",
        "kategorie": "Stark",
        "preis": 9.0,
        "spirituose": ["Dunkler Rum"],
        "geschmack": ["fruchtig", "tropisch"],
        "staerke": "hoch",
        "zutaten": ["Dunkler Rum", "Kokossirup", "Ananassaft"],
        "beschreibung": "Tropisch mit Kokos.",
    },
    {
        "name": "Gin Sour",
        "kategorie": "Klassisch",
        "preis": 8.0,
        "spirituose": ["Gin"],
        "geschmack": ["sauer"],
        "staerke": "mittel",
        "zutaten": ["Gin", "Zitronensaft"],
        "beschreibung": "Klassischer Sour.",
    },
]


class FakeRepository:
    def list_all(self):
        return COCKTAILS


class ChatServiceTest(unittest.TestCase):
    def test_extract_search_criteria_rejects_invalid_llm_output(self):
        async def fake_query_ollama(*args, **kwargs):
            return '{"spirituose": 123, "unbekannt": "x"}'

        with patch.object(chat_service, "query_ollama", fake_query_ollama):
            with self.assertRaises(chat_service.InvalidLLMOutputError):
                asyncio.run(chat_service.extract_search_criteria("Ich moechte Rum"))

    def test_chat_response_handles_invalid_llm_criteria(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            response = asyncio.run(chat_service.build_chat_response("Blabla", repository=FakeRepository()))

        self.assertEqual(response["cocktails"], [])
        self.assertEqual(response["type"], "follow_up")

    def test_chat_response_falls_back_for_fruity_without_coconut(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        async def fake_generate_answer(user_message, matching_cocktails, history=None):
            return {"message": "ok", "answer": "ok", "cocktails": matching_cocktails}

        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            with patch.object(chat_service, "generate_answer", fake_generate_answer):
                response = asyncio.run(
                    chat_service.build_chat_response(
                        "Ich haette gerne etwas fruchtiges ohne kokos",
                        repository=FakeRepository(),
                    )
                )

        self.assertEqual([cocktail["name"] for cocktail in response["cocktails"]], ["Caribbean Dream"])
        self.assertEqual(response["criteria"]["geschmack"], "fruchtig")
        self.assertIn("Kokossirup", response["criteria"]["ausschluesse"])

    def test_memory_combines_rum_and_disliked_sour(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        async def fake_generate_answer(user_message, matching_cocktails, history=None):
            return {"message": "ok", "answer": "ok", "cocktails": matching_cocktails}

        conversations = ConversationService()
        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            with patch.object(chat_service, "generate_answer", fake_generate_answer):
                asyncio.run(
                    chat_service.build_chat_response(
                        "Ich mag Rum.",
                        repository=FakeRepository(),
                        session_id="memory-1",
                        conversations=conversations,
                    )
                )
                response = asyncio.run(
                    chat_service.build_chat_response(
                        "Aber nichts Saures.",
                        repository=FakeRepository(),
                        session_id="memory-1",
                        conversations=conversations,
                    )
                )

        self.assertEqual(response["preferences"]["spirits"], ["Rum"])
        self.assertEqual(response["preferences"]["disliked_flavors"], ["sauer"])

    def test_memory_updates_strength_instead_of_accumulating(self):
        conversations = ConversationService()
        preferences = chat_service.local_update_preferences(
            conversations.get_preferences("memory-2"),
            "Ich moechte etwas Starkes.",
            COCKTAILS,
        )
        conversations.update_preferences("memory-2", preferences)

        preferences = chat_service.local_update_preferences(
            conversations.get_preferences("memory-2"),
            "Doch lieber etwas Leichtes.",
            COCKTAILS,
        )

        self.assertEqual(preferences.strength, "mild")

    def test_follow_up_for_empty_recommendation_request(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Empfiehl mir etwas.",
                    repository=FakeRepository(),
                    conversations=ConversationService(),
                )
            )

        self.assertEqual(response["type"], "follow_up")
        self.assertEqual(response["cocktails"], [])

    def test_recommendation_for_gin_preference(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        async def fake_generate_answer(user_message, matching_cocktails, history=None):
            return {"message": "ok", "answer": "ok", "cocktails": matching_cocktails}

        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            with patch.object(chat_service, "generate_answer", fake_generate_answer):
                response = asyncio.run(
                    chat_service.build_chat_response(
                        "Ich liebe Gin.",
                        repository=FakeRepository(),
                        conversations=ConversationService(),
                    )
                )

        self.assertEqual(response["type"], "recommendation")

    def test_later_dislike_removes_gin_from_positive_preferences(self):
        conversations = ConversationService()
        preferences = chat_service.local_update_preferences(
            conversations.get_preferences("memory-5"),
            "Ich liebe Gin.",
            COCKTAILS,
        )
        conversations.update_preferences("memory-5", preferences)

        preferences = chat_service.local_update_preferences(
            conversations.get_preferences("memory-5"),
            "Eigentlich doch keinen Gin.",
            COCKTAILS,
        )

        self.assertEqual(preferences.spirits, [])
        self.assertIn("Gin", preferences.disliked_ingredients)

    def test_answer_validation_rejects_unknown_cocktail_name(self):
        payload = {
            "answer": "Ich empfehle Zombie und Fantasie Drink.",
            "cocktail_names": ["Zombie", "Fantasie Drink"],
        }
        matching_cocktails = [{"name": "Zombie"}]

        with self.assertRaises(chat_service.InvalidLLMOutputError):
            chat_service.validate_answer_payload(payload, matching_cocktails)

    def test_normalize_search_criteria_infers_fruity_taste_from_message(self):
        criteria = CocktailSearchCriteria()

        normalized = chat_service.normalize_search_criteria(
            criteria,
            "Ich moechte einen fruchtigen Cocktail",
            COCKTAILS,
        )

        self.assertEqual(normalized.geschmack, "fruchtig")

    def test_normalize_search_criteria_canonicalizes_inflected_creamy_category(self):
        criteria = CocktailSearchCriteria(kategorie="cremigen Cocktail")

        normalized = chat_service.normalize_search_criteria(
            criteria,
            "Ich suche einen cremigen Cocktail",
            COCKTAILS,
        )

        self.assertEqual(normalized.kategorie, "Cremig")
        self.assertEqual(normalized.geschmack, "cremig")
