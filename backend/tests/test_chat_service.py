import asyncio
import os
import unittest
from unittest.mock import patch

from models.cocktail import CocktailSearchCriteria
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
        async def unavailable_intent_llm(*args, **kwargs):
            raise chat_service.LLMError("intent model unavailable in unit test")

        self.intent_patcher = patch.object(
            chat_service, "analyze_intent_with_llm", unavailable_intent_llm
        )
        self.intent_patcher.start()
        self.addCleanup(self.intent_patcher.stop)
        self.intent_answer_patcher = patch.object(
            chat_service, "generate_intent_answer", unavailable_intent_llm
        )
        self.intent_answer_patcher.start()
        self.addCleanup(self.intent_answer_patcher.stop)

    def test_extract_search_criteria_rejects_invalid_llm_output(self):
        async def fake_query_ollama(*args, **kwargs):
            return '{"spirituose": 123, "unbekannt": "x"}'

        with patch.object(chat_service, "query_ollama", fake_query_ollama):
            with self.assertRaises(chat_service.InvalidLLMOutputError):
                asyncio.run(chat_service.extract_search_criteria("Ich moechte Rum"))

    def test_chat_response_returns_unknown_for_unrecognized_message(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise AssertionError("Unknown messages should not call the LLM")

        with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
            response = asyncio.run(chat_service.build_chat_response("Blabla", repository=FakeRepository()))

        self.assertEqual(response["cocktails"], [])
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["intent"], "unknown")

    def test_greeting_uses_generated_llm_answer(self):
        async def generated_greeting(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="greeting", answer="Schön, dass du da bist. Worauf hast du Lust?"
            )

        with patch.object(chat_service, "analyze_intent_with_llm", generated_greeting):
            response = asyncio.run(
                chat_service.build_chat_response("Hallo!", repository=FakeRepository())
            )

        self.assertEqual(response["intent"], "greeting")
        self.assertEqual(response["answer"], "Schön, dass du da bist. Worauf hast du Lust?")
        self.assertEqual(response["cocktails"], [])

    def test_general_cocktail_statement_is_conversation_not_recommendation(self):
        async def generated_conversation(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="conversation",
                answer="Das klingt gut. Welche Geschmacksrichtung magst du?",
            )

        with patch.object(chat_service, "analyze_intent_with_llm", generated_conversation):
            response = asyncio.run(
                chat_service.build_chat_response("Ich mag Cocktails", repository=FakeRepository())
            )

        self.assertEqual(response["intent"], "conversation")
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["cocktails"], [])
        self.assertIn("Geschmacksrichtung", response["answer"])

    def test_local_fallback_treats_general_cocktail_statement_as_conversation(self):
        intent = chat_service.detect_intent(
            "Ich mag Cocktails",
            chat_service.CocktailPreferences(),
            chat_service.CocktailPreferences(),
            COCKTAILS,
        )

        self.assertEqual(intent, "conversation")

    def test_conversation_answer_requires_a_follow_up_question(self):
        incomplete = chat_service.IntentAnalysis(
            intent="conversation", answer="Cocktails sind wirklich vielseitig."
        )
        complete = chat_service.IntentAnalysis(
            intent="conversation", answer="Cocktails sind vielseitig. Was magst du besonders gern?"
        )

        self.assertFalse(chat_service.intent_answer_is_usable(incomplete))
        self.assertTrue(chat_service.intent_answer_is_usable(complete))

    def test_follow_up_reply_cannot_be_classified_as_greeting(self):
        analysis = chat_service.IntentAnalysis(
            intent="greeting", answer="Hallo! Wie geht es dir heute?"
        )
        history = [
            chat_service.ChatMessage(role="assistant", content="Hallo! Wie geht es dir heute?")
        ]

        self.assertFalse(
            chat_service.intent_answer_is_usable(analysis, "Gut und dir?", history)
        )
        self.assertEqual(
            chat_service.detect_intent(
                "Gut und dir?",
                chat_service.CocktailPreferences(),
                chat_service.CocktailPreferences(),
                COCKTAILS,
            ),
            "conversation",
        )

    def test_follow_up_reply_uses_dynamic_answer_after_classifier_fallback(self):
        async def dynamic_answer(*args, **kwargs):
            return "Mir geht es gut, danke. Welche Geschmacksrichtung magst du bei Cocktails?"

        with patch.object(chat_service, "generate_intent_answer", dynamic_answer):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Gut und dir?",
                    history=[
                        chat_service.ChatMessage(
                            role="assistant", content="Hallo! Wie geht es dir heute?"
                        )
                    ],
                    repository=FakeRepository(),
                )
            )

        self.assertEqual(response["intent"], "conversation")
        self.assertEqual(
            response["answer"],
            "Mir geht es gut, danke. Welche Geschmacksrichtung magst du bei Cocktails?",
        )

    def test_short_rejection_does_not_match_honey_or_trigger_recommendation(self):
        cocktails = COCKTAILS + [{
            "name": "Honey Drink",
            "preis": 8.0,
            "spirituose": ["Jack Daniel's Honey"],
            "geschmack": ["süß"],
            "staerke": "mittel",
            "zutaten": ["Jack Daniel's Honey"],
            "beschreibung": "Ein Whiskey-Drink.",
        }]
        preferences = chat_service.local_update_preferences(
            chat_service.CocktailPreferences(), "ne", cocktails
        )

        self.assertEqual(preferences.spirits, [])
        self.assertEqual(
            chat_service.detect_intent(
                "ne", chat_service.CocktailPreferences(), preferences, cocktails
            ),
            "conversation",
        )

    def test_repeated_assistant_answer_is_rejected(self):
        history = [
            chat_service.ChatMessage(
                role="assistant",
                content="Hallo! Ich bin CocktailGPT. Welche Cocktails magst du?",
            )
        ]
        repeated = chat_service.IntentAnalysis(
            intent="conversation",
            answer="Hallo, ich bin CocktailGPT. Welche Cocktails magst du?",
        )

        self.assertFalse(
            chat_service.intent_answer_is_usable(repeated, "Gut und dir?", history)
        )

    def test_out_of_scope_question_gets_boundary_response(self):
        response = asyncio.run(
            chat_service.build_chat_response("Wie wird das Wetter?", repository=FakeRepository())
        )

        self.assertEqual(response["intent"], "out_of_scope")
        self.assertIn("nicht zuständig", response["answer"])

    def test_cocktail_detail_returns_only_requested_information(self):
        response = asyncio.run(
            chat_service.build_chat_response("Was kostet der Gin Sour?", repository=FakeRepository())
        )

        self.assertEqual(response["intent"], "cocktail_details")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertIn("8.00 Euro", response["answer"])

    def test_strong_preference_is_not_misclassified_as_detail_question(self):
        preferences = chat_service.local_update_preferences(
            chat_service.CocktailPreferences(), "Ich möchte etwas Starkes", COCKTAILS
        )

        intent = chat_service.detect_intent(
            "Ich möchte etwas Starkes", chat_service.CocktailPreferences(), preferences, COCKTAILS
        )

        self.assertEqual(intent, "preference_update")

    def test_strong_and_creamy_only_store_explicit_preferences(self):
        async def preference_intent(*args, **kwargs):
            return chat_service.IntentAnalysis(intent="preference_update", answer="")

        async def fake_generate_answer(user_message, matching_cocktails, history=None):
            return {"message": "ok", "answer": "ok", "cocktails": matching_cocktails}

        with patch.object(chat_service, "analyze_intent_with_llm", preference_intent):
            with patch.object(chat_service, "generate_answer", fake_generate_answer):
                response = asyncio.run(
                    chat_service.build_chat_response(
                        "Ich mag starke und cremige Cocktails", repository=FakeRepository()
                    )
                )

        self.assertEqual(response["preferences"]["liked_flavors"], ["cremig"])
        self.assertEqual(response["preferences"]["strength"], "stark")
        self.assertEqual(response["preferences"]["liked_ingredients"], [])
        self.assertEqual(response["preferences"]["spirits"], [])
        self.assertIsNone(response["preferences"]["alcoholic"])

    def test_reset_intent_clears_session_preferences(self):
        conversations = ConversationService()
        conversations.update_preferences(
            "reset-me",
            chat_service.local_update_preferences(
                conversations.get_preferences("reset-me"), "Ich mag Gin", COCKTAILS
            ),
        )

        response = asyncio.run(
            chat_service.build_chat_response(
                "Bitte alles zurücksetzen",
                repository=FakeRepository(),
                session_id="reset-me",
                conversations=conversations,
            )
        )

        self.assertEqual(response["intent"], "reset_preferences")
        self.assertEqual(conversations.get_preferences("reset-me").spirits, [])

    def test_intent_routing_can_be_disabled(self):
        async def fake_update_preferences_with_llm(message, preferences, cocktails):
            raise chat_service.InvalidLLMOutputError("bad output")

        with patch.dict(os.environ, {"INTENT_ROUTING_ENABLED": "false"}):
            with patch.object(chat_service, "update_preferences_with_llm", fake_update_preferences_with_llm):
                response = asyncio.run(
                    chat_service.build_chat_response("Hallo", repository=FakeRepository())
                )

        self.assertIsNone(response["intent"])
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

    def test_fruity_preference_only_sets_fruity_flavor(self):
        preferences = chat_service.local_update_preferences(
            chat_service.CocktailPreferences(),
            "Ich mag fruchtige Cocktails",
            COCKTAILS,
        )

        self.assertEqual(preferences.liked_flavors, ["fruchtig"])
        self.assertEqual(preferences.spirits, [])
        self.assertEqual(preferences.liked_ingredients, [])
        self.assertEqual(preferences.disliked_ingredients, [])
        self.assertIsNone(preferences.strength)
        self.assertIsNone(preferences.alcoholic)

    def test_llm_preferences_are_limited_to_explicit_local_signal(self):
        current = chat_service.CocktailPreferences()
        local = chat_service.local_update_preferences(current, "Ich mag fruchtige Cocktails", COCKTAILS)
        llm = chat_service.CocktailPreferences(
            liked_ingredients=["Rum", "Ananassaft"],
            spirits=["Rum"],
            liked_flavors=["fruchtig", "cremig"],
            strength="mittel",
            alcoholic=True,
        )

        constrained = chat_service.constrain_preferences_to_local_signal(current, llm, local)

        self.assertEqual(constrained.liked_flavors, ["fruchtig"])
        self.assertEqual(constrained.spirits, [])
        self.assertEqual(constrained.liked_ingredients, [])
        self.assertIsNone(constrained.strength)
        self.assertIsNone(constrained.alcoholic)

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
