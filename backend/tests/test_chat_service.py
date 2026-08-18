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

    def test_scope_llm_routes_emotional_statement_out_of_scope(self):
        async def fake_query(*args, **kwargs):
            return (
                '{"scope":"out_of_scope","answer":"Dabei kann ich dir nicht inhaltlich helfen. '
                'Suchst du stattdessen einen Cocktail oder ein Rezept?","confidence":0.98,'
                '"spelling_errors":[],"eligible_word_count":4}'
            )

        with patch.object(chat_service, "query_ollama", fake_query):
            result = asyncio.run(
                chat_service.analyze_intent_with_llm(
                    "Ich bin sehr traurig", None, COCKTAILS
                )
            )

        self.assertEqual(result.intent, "out_of_scope")
        self.assertEqual(result.action, "reject")
        self.assertIn("Cocktail", result.answer)

    def test_scope_llm_allows_cocktail_request_with_emotional_wording(self):
        responses = iter([
            (
                '{"scope":"cocktail","answer":"","confidence":0.97,'
                '"spelling_errors":[],"eligible_word_count":7}'
            ),
            (
                '{"intent":"recommendation","action":"recommend","cocktail_name":null,'
                '"attribute":"flavor","value":"fruchtig","context_mode":"new_query",'
                '"confidence":0.96,"answer":""}'
            ),
        ])

        async def fake_query(*args, **kwargs):
            return next(responses)

        with patch.object(chat_service, "query_ollama", fake_query):
            result = asyncio.run(
                chat_service.analyze_intent_with_llm(
                    "Empfiehl meiner traurigen Freundin einen fruchtigen Cocktail",
                    None,
                    COCKTAILS,
                )
            )

        self.assertEqual(result.intent, "recommendation")

    def test_scope_requires_explicit_spelling_analysis(self):
        async def fake_query(*args, **kwargs):
            return '{"scope":"cocktail","answer":"","confidence":0.97}'

        with patch.object(chat_service, "query_ollama", fake_query):
            with self.assertRaises(chat_service.InvalidLLMOutputError):
                asyncio.run(chat_service.analyze_scope_with_llm("Hallo", None))

    def test_backend_calculates_spelling_ratio_from_llm_error_list(self):
        responses = iter([
            (
                '{"scope":"cocktail","answer":"","confidence":0.97,'
                '"spelling_errors":["möchhte","fruchtign","coktail"],"eligible_word_count":5}'
            ),
            (
                '{"intent":"recommendation","action":"recommend","cocktail_name":null,'
                '"attribute":"flavor","value":"fruchtig","context_mode":"new_query",'
                '"confidence":0.96,"answer":""}'
            ),
        ])

        async def fake_query(*args, **kwargs):
            return next(responses)

        with patch.object(chat_service, "query_ollama", fake_query):
            result = asyncio.run(
                chat_service.analyze_intent_with_llm(
                    "ich möchhte ein fruchtign coktail", None, COCKTAILS
                )
            )

        self.assertEqual(result.spelling_error_ratio, 0.6)

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

    def test_personal_support_answer_is_rejected_even_as_conversation(self):
        unsafe_answer = IntentAnalysis(
            intent="conversation",
            action="respond",
            answer=(
                "Das tut mir sehr leid. Ich bin hier, um dir zuzuhören. "
                "Wie geht es dir in diesem Moment?"
            ),
        )

        self.assertFalse(
            chat_service.intent_answer_is_usable(
                unsafe_answer, "Mein Haustier ist gestorben"
            )
        )

    def test_valid_llm_intent_is_not_overridden_by_local_rules(self):
        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="conversation",
                action="respond",
                answer="Dabei kann ich nicht helfen. Welchen Cocktailgeschmack magst du?",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ich habe Probleme mit meiner Freundin, was kann ich tun?",
                    repository=FakeRepository(),
                )
            )

        self.assertEqual(response["intent"], "conversation")
        self.assertEqual(
            response["answer"],
            "Dabei kann ich nicht helfen. Welchen Cocktailgeschmack magst du?",
        )
        self.assertEqual(response["cocktails"], [])

    def test_llm_unknown_message_does_not_start_database_search(self):
        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="unknown",
                action="clarify",
                answer="Suchst du eine Empfehlung oder Informationen zu einem Cocktail?",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response("Blabla", repository=FakeRepository())
            )

        self.assertEqual(response["intent"], "unknown")
        self.assertEqual(response["cocktails"], [])

    def test_llm_intent_failure_does_not_guess_with_local_rules(self):
        async def unavailable(*args, **kwargs):
            raise chat_service.LLMError("LLM unavailable in unit test")

        with patch.object(chat_service, "analyze_intent_with_llm", unavailable):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Empfiehl mir etwas Cremiges", repository=FakeRepository()
                )
            )

        self.assertEqual(response["type"], "follow_up")
        self.assertEqual(response["intent"], "unknown")
        self.assertEqual(response["cocktails"], [])

    def test_spelling_error_ratio_above_35_percent_blocks_ordering(self):
        conversations = ConversationService()

        async def analyze(*args, **kwargs):
            return IntentAnalysis(
                intent="recommendation",
                action="recommend",
                context_mode="new_query",
                spelling_error_ratio=0.36,
                eligible_word_count=10,
            )

        with patch.object(chat_service, "analyze_intent_with_llm", analyze):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Falsch geschriebene Testanfrage",
                    repository=FakeRepository(),
                    conversations=conversations,
                    session_id="blocked-session",
                )
            )

        self.assertEqual(response["type"], "follow_up")
        self.assertTrue(conversations.is_ordering_blocked("blocked-session"))

    def test_ordering_block_survives_chat_reset_until_staff_unlocks(self):
        conversations = ConversationService()
        conversations.block_ordering("blocked-session")

        conversations.reset_session("blocked-session")

        self.assertTrue(conversations.is_ordering_blocked("blocked-session"))
        self.assertFalse(conversations.unlock_ordering("blocked-session", "0000"))
        self.assertTrue(conversations.is_ordering_blocked("blocked-session"))
        self.assertTrue(conversations.unlock_ordering("blocked-session", "1111"))
        self.assertFalse(conversations.is_ordering_blocked("blocked-session"))

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

    def test_excluded_spirit_is_not_reintroduced_as_positive_criterion(self):
        criteria = chat_service.normalize_search_criteria(
            CocktailSearchCriteria(
                geschmack="fruchtig",
                ausschluesse=["Rum"],
            ),
            "einen fruchtigen Cocktail ohne Rum",
            COCKTAILS,
        )

        self.assertIsNone(criteria.spirituose)
        self.assertEqual(criteria.geschmack, "fruchtig")
        self.assertEqual(criteria.ausschluesse, ["Rum"])

    def test_fruity_cocktail_without_rum_can_return_non_rum_match(self):
        cocktails = COCKTAILS + [{
            "name": "Vodka Berry",
            "preis": 8.0,
            "spirituose": ["Vodka"],
            "geschmack": ["fruchtig"],
            "staerke": "mittel",
            "zutaten": ["Vodka", "Beerensaft"],
            "beschreibung": "Beerig und frisch.",
        }]
        criteria = chat_service.normalize_search_criteria(
            CocktailSearchCriteria(
                geschmack="fruchtig",
                ausschluesse=["Rum"],
            ),
            "einen fruchtigen Cocktail ohne Rum",
            cocktails,
        )

        matches = chat_service.search_cocktails(criteria, cocktails, limit=10)

        self.assertEqual([cocktail["name"] for cocktail in matches], ["Vodka Berry"])

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
