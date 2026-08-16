import asyncio
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from models.cocktail import ChatResponse, CocktailSearchCriteria, IntentAnalysis
from services import chat_service
from services.conversation_service import ConversationService


COCKTAILS = [
    {
        "name": "Caribbean Dream",
        "preis": 8.5,
        "spirituose": ["Rum"],
        "geschmack": ["cremig", "fruchtig"],
        "staerke": "mittel",
        "zutaten": ["Rum", "Ananassaft", "Sahne"],
        "beschreibung": "Cremig und fruchtig.",
    },
    {
        "name": "Bahama Mama",
        "preis": 9.0,
        "spirituose": ["Dunkler Rum"],
        "geschmack": ["fruchtig", "tropisch", "stark"],
        "staerke": "hoch",
        "zutaten": ["Dunkler Rum", "Kokossirup", "Ananassaft"],
        "beschreibung": "Tropisch mit Kokos.",
    },
    {
        "name": "Gin Sour",
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
    def setUp(self):
        async def unavailable(*args, **kwargs):
            raise chat_service.LLMError("LLM unavailable in unit test")

        self.intent_patcher = patch.object(
            chat_service, "analyze_intent_with_llm", unavailable
        )
        self.intent_patcher.start()
        self.addCleanup(self.intent_patcher.stop)
        self.answer_patcher = patch.object(
            chat_service, "generate_intent_answer", unavailable
        )
        self.answer_patcher.start()
        self.addCleanup(self.answer_patcher.stop)

    def test_removed_intents_are_rejected(self):
        with self.assertRaises(ValidationError):
            IntentAnalysis(intent="preference_update")
        with self.assertRaises(ValidationError):
            IntentAnalysis(intent="reset_preferences")

    def test_response_model_has_no_remembered_wishes(self):
        response = ChatResponse(message="ok", answer="ok")
        self.assertNotIn("preferences", ChatResponse.model_fields)

    def test_conversation_service_has_no_preference_store(self):
        conversations = ConversationService()
        self.assertFalse(hasattr(conversations, "get_preferences"))
        self.assertFalse(hasattr(conversations, "update_preferences"))

    def test_extract_search_criteria_rejects_invalid_llm_output(self):
        async def fake_query(*args, **kwargs):
            return '{"spirituose": 123, "unbekannt": "x"}'

        with patch.object(chat_service, "query_ollama", fake_query):
            with self.assertRaises(chat_service.InvalidLLMOutputError):
                asyncio.run(chat_service.extract_search_criteria("mit Rum"))

    def test_local_intent_detection(self):
        cases = {
            "Hallo": "conversation",
            "Wie ist das Wetter?": "out_of_scope",
            "Ich habe Probleme in meiner Beziehung": "out_of_scope",
            "Empfiehl mir etwas Cremiges": "recommendation",
            "Ueberrasch mich": "random",
            "Habt ihr Gin Sour?": "catalog_query",
            "Blabla": "unknown",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(chat_service.detect_intent(message, COCKTAILS), expected)

    def test_general_cocktail_statement_is_conversation(self):
        self.assertEqual(
            chat_service.detect_intent("Ich mag Cocktails", COCKTAILS),
            "conversation",
        )

    def test_greeting_uses_conversation_response(self):
        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="conversation",
                action="respond",
                answer="Hallo! Welche Geschmacksrichtung magst du?",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response("Hallo", repository=FakeRepository())
            )

        self.assertEqual(response["intent"], "conversation")
        self.assertEqual(response["cocktails"], [])
        self.assertNotIn("preferences", response)

    def test_out_of_scope_answer_requires_cocktail_transition(self):
        advice = IntentAnalysis(
            intent="out_of_scope",
            action="reject",
            answer="Du solltest offen mit deiner Partnerin sprechen und ihr gut zuhören.",
        )
        redirect = IntentAnalysis(
            intent="out_of_scope",
            action="reject",
            answer=(
                "Bei Beziehungsfragen kann ich dir keine Ratschläge geben. "
                "Suchst du stattdessen einen Cocktail für einen entspannten Abend?"
            ),
        )

        self.assertFalse(
            chat_service.intent_answer_is_usable(
                advice, "Ich habe Probleme in meiner Beziehung"
            )
        )
        self.assertTrue(
            chat_service.intent_answer_is_usable(
                redirect, "Ich habe Probleme in meiner Beziehung"
            )
        )

    def test_unknown_message_does_not_start_database_search(self):
        response = asyncio.run(
            chat_service.build_chat_response("Blabla", repository=FakeRepository())
        )
        self.assertEqual(response["intent"], "unknown")
        self.assertEqual(response["cocktails"], [])

    def test_low_confidence_interpretation_asks_for_clarification(self):
        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="unknown",
                action="clarify",
                confidence=0.4,
                answer="Welchen Cocktail soll ich prüfen?",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response("Ist der gut?", repository=FakeRepository())
            )

        self.assertEqual(response["type"], "follow_up")
        self.assertEqual(response["cocktails"], [])

    def test_llm_catalog_plan_is_executed_against_database(self):
        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="catalog_query",
                action="check_attribute",
                cocktail_name="Gin Sour",
                attribute="ingredients",
                context_mode="new_query",
            )

        async def grounded(*args, **kwargs):
            return "Gin Sour enthält Gin und Zitronensaft."

        with (
            patch.object(chat_service, "analyze_intent_with_llm", analyze),
            patch.object(chat_service, "generate_grounded_catalog_answer", grounded),
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Was ist im Gin Sour?", repository=FakeRepository()
                )
            )

        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertIn("Zitronensaft", response["answer"])

    def test_previous_cocktail_reference_is_structured_context(self):
        conversations = ConversationService()
        conversations.set_referenced_cocktail("session", "Gin Sour")

        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="catalog_query",
                action="check_attribute",
                cocktail_name="Gin Sour",
                attribute="flavor",
                context_mode="previous_cocktail",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ist der sauer?",
                    repository=FakeRepository(),
                    conversations=conversations,
                    session_id="session",
                )
            )

        self.assertEqual(response["cocktails"][0]["name"], "Gin Sour")

    def test_reset_session_clears_only_conversation_context(self):
        conversations = ConversationService()
        conversations.set_referenced_cocktail("session", "Gin Sour")
        conversations.update_last_random_cocktail("session", "Bahama Mama")
        conversations.reset_session("session")
        self.assertIsNone(conversations.get_catalog_context("session").referenced_cocktail)
        self.assertIsNone(conversations.get_last_random_cocktail("session"))

    def test_recommendation_uses_only_current_message_criteria(self):
        conversations = ConversationService()

        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="recommendation",
                action="recommend",
                context_mode="new_query",
            )

        async def extract(message):
            if "Rum" in message:
                return CocktailSearchCriteria(spirituose="Rum")
            return CocktailSearchCriteria(geschmack="sauer")

        async def fail_answer(*args, **kwargs):
            raise chat_service.LLMError("use deterministic fallback")

        with (
            patch.object(chat_service, "analyze_intent_with_llm", analyze),
            patch.object(chat_service, "extract_search_criteria", extract),
            patch.object(chat_service, "generate_answer", fail_answer),
        ):
            first = asyncio.run(
                chat_service.build_chat_response(
                    "Empfiehl mir etwas mit Rum",
                    repository=FakeRepository(),
                    conversations=conversations,
                    session_id="same-session",
                )
            )
            second = asyncio.run(
                chat_service.build_chat_response(
                    "Empfiehl mir etwas Saures",
                    repository=FakeRepository(),
                    conversations=conversations,
                    session_id="same-session",
                )
            )

        self.assertEqual(first["criteria"]["spirituose"], "Rum")
        self.assertIsNone(second["criteria"]["spirituose"])
        self.assertEqual(second["criteria"]["geschmack"], "sauer")
        self.assertEqual([item["name"] for item in second["cocktails"]], ["Gin Sour"])
        self.assertNotIn("preferences", first)
        self.assertNotIn("preferences", second)

    def test_llm_criteria_are_normalized_to_catalog_values(self):
        criteria = chat_service.normalize_search_criteria(
            CocktailSearchCriteria(geschmack="fruchtig"), "", COCKTAILS
        )
        self.assertEqual(criteria.geschmack, "fruchtig")

    def test_recommendation_randomly_selects_three_from_all_matches(self):
        criteria = CocktailSearchCriteria()
        expected = [COCKTAILS[2], COCKTAILS[0]]
        with patch.object(chat_service.random, "sample", return_value=expected) as sample:
            selected = chat_service.select_recommendation_candidates(
                criteria, COCKTAILS, limit=2
            )

        self.assertEqual(selected, expected)
        self.assertEqual(len(sample.call_args.args[0]), len(COCKTAILS))
        sample.assert_called_once()

    def test_local_criteria_fallback_extracts_exclusion(self):
        criteria = chat_service.fallback_search_criteria(
            "Etwas fruchtiges ohne Kokos", COCKTAILS
        )
        self.assertEqual(criteria.geschmack, "fruchtig")
        self.assertIn("Kokossirup", criteria.ausschluesse)

    def test_random_response_uses_catalog_and_avoids_immediate_repeat(self):
        conversations = ConversationService()
        conversations.update_last_random_cocktail("session", "Caribbean Dream")
        with patch.object(chat_service.random, "choice", side_effect=lambda values: values[0]):
            response = chat_service.build_random_response(
                COCKTAILS, "session", conversations
            )

        self.assertNotEqual(response["selected_cocktail"]["name"], "Caribbean Dream")
        self.assertIsNone(response["criteria"])
        self.assertNotIn("preferences", response)

    def test_answer_validation_rejects_unknown_cocktail_name(self):
        payload = {"answer": "Nimm Mojito", "cocktail_names": ["Mojito"]}
        with self.assertRaises(chat_service.InvalidLLMOutputError):
            chat_service.validate_answer_payload(payload, COCKTAILS)

    def test_generated_recommendation_uses_only_selected_database_rows(self):
        async def fake_query(*args, **kwargs):
            return '{"answer":"freie Antwort","cocktail_names":["Gin Sour"]}'

        with patch.object(chat_service, "query_ollama", fake_query):
            response = asyncio.run(
                chat_service.generate_answer("etwas saures", [COCKTAILS[2]])
            )

        self.assertEqual(response["cocktails"], [COCKTAILS[2]])
        self.assertIn("Gin Sour", response["answer"])
        self.assertNotIn("freie Antwort", response["answer"])


if __name__ == "__main__":
    unittest.main()
