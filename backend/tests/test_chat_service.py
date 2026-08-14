import asyncio
import json
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

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

    def test_greeting_is_handled_as_conversation(self):
        async def generated_conversation(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="conversation", answer="Schön, dass du da bist. Worauf hast du Lust?"
            )

        with patch.object(chat_service, "analyze_intent_with_llm", generated_conversation):
            response = asyncio.run(
                chat_service.build_chat_response("Hallo!", repository=FakeRepository())
            )

        self.assertEqual(response["intent"], "conversation")
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

    def test_random_intent_is_detected_locally(self):
        for message in [
            "Ueberrasch mich",
            "Zufaelliger Cocktail",
            "Ich kann mich nicht entscheiden",
        ]:
            with self.subTest(message=message):
                intent = chat_service.detect_intent(
                    message,
                    chat_service.CocktailPreferences(),
                    chat_service.CocktailPreferences(),
                    COCKTAILS,
                )

                self.assertEqual(intent, "random")

    def test_conversation_answer_requires_a_follow_up_question(self):
        incomplete = chat_service.IntentAnalysis(
            intent="conversation", answer="Cocktails sind wirklich vielseitig."
        )
        complete = chat_service.IntentAnalysis(
            intent="conversation", answer="Cocktails sind vielseitig. Was magst du besonders gern?"
        )

        self.assertFalse(chat_service.intent_answer_is_usable(incomplete))
        self.assertTrue(chat_service.intent_answer_is_usable(complete))

    def test_greeting_conversation_does_not_ask_about_personal_wellbeing(self):
        analysis = chat_service.IntentAnalysis(
            intent="conversation", answer="Hallo! Wie geht es dir heute?"
        )

        self.assertFalse(
            chat_service.intent_answer_is_usable(analysis, "Hallo!", [])
        )
        self.assertEqual(
            chat_service.detect_intent(
                "Hallo!",
                chat_service.CocktailPreferences(),
                chat_service.CocktailPreferences(),
                COCKTAILS,
            ),
            "conversation",
        )

    def test_removed_intents_are_rejected(self):
        for removed_intent in {"greeting", "cocktail_details"}:
            with self.subTest(intent=removed_intent):
                with self.assertRaises(ValidationError):
                    chat_service.IntentAnalysis(intent=removed_intent, answer="")

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

    def test_conversation_rejects_unchecked_cocktail_recommendation(self):
        analysis = chat_service.IntentAnalysis(
            intent="conversation",
            answer="Wie wäre es mit einem erfundenen Super Drink?",
        )

        self.assertFalse(chat_service.intent_answer_is_usable(analysis, "ne", []))

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

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertIn("8.00 Euro", response["answer"])

    def test_mojito_catalog_query_returns_regular_and_virgin_variants(self):
        cocktails = COCKTAILS + [
            {
                "name": "Mojito",
                "preis": 7.5,
                "spirituose": ["Rum"],
                "geschmack": ["frisch"],
                "staerke": "mittel",
                "zutaten": ["Rum", "Limette", "Minze"],
                "beschreibung": "Klassischer Mojito.",
            },
            {
                "name": "Virgin Mojito",
                "preis": 5.9,
                "spirituose": [],
                "geschmack": ["frisch"],
                "staerke": "alkoholfrei",
                "zutaten": ["Limette", "Minze", "Soda"],
                "beschreibung": "Alkoholfreier Mojito.",
            },
        ]

        response = chat_service.build_catalog_query_response(
            "Habt ihr Mojitos?", cocktails, chat_service.CocktailPreferences()
        )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual(
            [item["name"] for item in response["cocktails"]],
            ["Mojito", "Virgin Mojito"],
        )
        self.assertIn("Mojito, Virgin Mojito", response["answer"])

    def test_local_intent_detects_catalog_query(self):
        intent = chat_service.detect_intent(
            "Habt ihr Mojitos?",
            chat_service.CocktailPreferences(),
            chat_service.CocktailPreferences(),
            COCKTAILS,
        )

        self.assertEqual(intent, "catalog_query")

    def test_catalog_query_uses_llm_interpretation_and_keeps_preferences(self):
        repository = FakeRepository()
        conversations = ConversationService()

        async def interpreted_catalog_query(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="catalog_query",
                action="check_availability",
                cocktail_name="Gin Sour",
                attribute="availability",
                confidence=0.98,
            )

        with patch.object(
            chat_service,
            "analyze_intent_with_llm",
            interpreted_catalog_query,
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Habt ihr einen Gin Sour?",
                    repository=repository,
                    session_id="catalog-session",
                    conversations=conversations,
                )
            )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertEqual(response["preferences"], chat_service.CocktailPreferences().model_dump())

    def test_catalog_query_by_spirit_does_not_store_preferences(self):
        response = chat_service.build_catalog_query_response(
            "Welche Cocktails enthalten Gin?", COCKTAILS, chat_service.CocktailPreferences()
        )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertEqual(response["preferences"]["spirits"], [])

    def test_exact_mojito_detail_prefers_regular_mojito(self):
        cocktails = COCKTAILS + [
            {**COCKTAILS[0], "name": "Mojito", "preis": 7.5},
            {**COCKTAILS[0], "name": "Virgin Mojito", "preis": 5.9},
        ]

        response = chat_service.build_cocktail_detail_response(
            "Was kostet der Mojito?", cocktails, chat_service.CocktailPreferences()
        )

        self.assertEqual([item["name"] for item in response["cocktails"]], ["Mojito"])
        self.assertIn("7.50 Euro", response["answer"])

    def test_compact_cocktail_name_flavor_question_returns_database_fact(self):
        pina_colada = {
            **COCKTAILS[0],
            "name": "Pina Colada",
            "geschmack": ["süß", "fruchtig", "cremig"],
        }
        cocktails = COCKTAILS + [pina_colada]

        intent = chat_service.detect_intent(
            "Ist der pinacolada cremig?",
            chat_service.CocktailPreferences(),
            chat_service.local_update_preferences(
                chat_service.CocktailPreferences(), "Ist der pinacolada cremig?", cocktails
            ),
            cocktails,
        )
        response = chat_service.build_cocktail_detail_response(
            "Ist der pinacolada cremig?", cocktails, chat_service.CocktailPreferences()
        )

        self.assertEqual(intent, "catalog_query")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Pina Colada"])
        self.assertEqual(response["answer"], "Ja, Pina Colada ist laut unserer Karte cremig.")
        self.assertEqual(response["preferences"]["liked_flavors"], [])

    def test_typo_name_does_not_match_cocktail_via_common_word(self):
        cocktails = COCKTAILS + [
            {
                **COCKTAILS[0],
                "name": "Pina Colada",
                "geschmack": ["süß", "fruchtig", "cremig"],
            },
            {
                **COCKTAILS[2],
                "name": "Rotkäppchen ist Sauer",
                "geschmack": ["fruchtig", "sauer"],
            },
        ]

        response = chat_service.build_catalog_query_response(
            "Ist der pinacolade cremig?", cocktails, chat_service.CocktailPreferences()
        )

        self.assertEqual([item["name"] for item in response["cocktails"]], ["Pina Colada"])
        self.assertEqual(response["answer"], "Ja, Pina Colada ist laut unserer Karte cremig.")

    def test_flavor_fact_question_uses_llm_plan_and_does_not_store_flavor(self):
        pina_colada = {
            **COCKTAILS[0],
            "name": "Pina Colada",
            "geschmack": ["süß", "fruchtig", "cremig"],
        }

        class PinaRepository:
            def list_all(self):
                return COCKTAILS + [pina_colada]

        async def interpreted_flavor_query(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="catalog_query",
                action="check_attribute",
                cocktail_name="Pina Colada",
                attribute="flavor",
                value="cremig",
                confidence=0.99,
            )

        with patch.object(
            chat_service,
            "analyze_intent_with_llm",
            interpreted_flavor_query,
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ist der pinacolada cremig?",
                    repository=PinaRepository(),
                    session_id="pina-fact",
                    conversations=ConversationService(),
                )
            )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertIn("Ja, Pina Colada", response["answer"])
        self.assertEqual(response["preferences"]["liked_flavors"], [])

    def test_low_confidence_interpretation_asks_instead_of_guessing(self):
        async def uncertain_interpretation(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="unknown",
                action="clarify",
                confidence=0.31,
                answer="Meinst du Pina Colada oder einen anderen Cocktail?",
            )

        with patch.object(
            chat_service, "analyze_intent_with_llm", uncertain_interpretation
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ist der Colada cremig?", repository=FakeRepository()
                )
            )

        self.assertEqual(response["type"], "follow_up")
        self.assertEqual(response["intent"], "unknown")
        self.assertEqual(
            response["answer"], "Meinst du Pina Colada oder einen anderen Cocktail?"
        )

    def test_catalog_flavor_request_does_not_reuse_previous_cocktail(self):
        async def misclassified_catalog_list(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="catalog_query",
                action="check_attribute",
                attribute="flavor",
                value="cremig",
                context_mode="new_query",
                confidence=0.8,
            )

        history = [
            chat_service.ChatMessage(role="user", content="Ist der Gin Sour auf der Karte?"),
            chat_service.ChatMessage(role="assistant", content="Ja, Gin Sour ist auf der Karte."),
        ]
        with patch.object(
            chat_service, "analyze_intent_with_llm", misclassified_catalog_list
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Okay, und habt ihr auch etwas Cremiges?",
                    history=history,
                    repository=FakeRepository(),
                )
            )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual(
            [cocktail["name"] for cocktail in response["cocktails"]],
            ["Caribbean Dream"],
        )
        self.assertNotIn("Gin Sour ist", response["answer"])

    def test_invalid_llm_attribute_is_inferred_from_catalog_value(self):
        sanitized = chat_service.sanitize_interpretation_payload(
            {
                "intent": "catalog_query",
                "action": "check_attribute",
                "attribute": "value",
                "value": "cremig",
                "context_mode": "new_search",
                "confidence": 0.8,
                "answer": "",
            },
            COCKTAILS,
        )

        self.assertEqual(sanitized["attribute"], "flavor")
        self.assertEqual(sanitized["context_mode"], "new_query")

    def test_previous_cocktail_mode_resolves_reference_from_history(self):
        async def previous_cocktail_query(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="catalog_query",
                action="check_attribute",
                attribute="flavor",
                value="sauer",
                context_mode="previous_cocktail",
                confidence=0.94,
            )

        history = [
            chat_service.ChatMessage(role="user", content="Habt ihr einen Gin Sour?"),
            chat_service.ChatMessage(role="assistant", content="Ja, Gin Sour ist auf der Karte."),
        ]
        with patch.object(
            chat_service, "analyze_intent_with_llm", previous_cocktail_query
        ):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ist der auch sauer?", history=history, repository=FakeRepository()
                )
            )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])
        self.assertIn("Ja, Gin Sour", response["answer"])

    def test_llm_first_recommendation_cannot_infer_unmentioned_ingredient(self):
        async def recommendation_interpretation(*args, **kwargs):
            return chat_service.IntentAnalysis(
                intent="recommendation",
                action="recommend",
                value="cremig",
                confidence=0.96,
            )

        async def inferred_preferences(*args, **kwargs):
            return chat_service.CocktailPreferences(
                liked_flavors=["cremig"], liked_ingredients=["Sahne"]
            )

        async def grounded_answer(user_message, matching_cocktails, history=None):
            return {"message": "ok", "answer": "ok", "cocktails": matching_cocktails}

        with patch.object(
            chat_service, "analyze_intent_with_llm", recommendation_interpretation
        ):
            with patch.object(
                chat_service, "update_preferences_with_llm", inferred_preferences
            ):
                with patch.object(chat_service, "generate_answer", grounded_answer):
                    response = asyncio.run(
                        chat_service.build_chat_response(
                            "Empfiehl mir etwas Cremiges", repository=FakeRepository()
                        )
                    )

        self.assertEqual(response["intent"], "recommendation")
        self.assertEqual(response["preferences"]["liked_flavors"], ["cremig"])
        self.assertEqual(response["preferences"]["liked_ingredients"], [])

    def test_polite_request_does_not_match_bitter_flavor(self):
        cocktails = COCKTAILS + [{
            **COCKTAILS[0],
            "name": "Bitter Drink",
            "geschmack": ["leicht bitter"],
        }]

        preferences = chat_service.local_update_preferences(
            chat_service.CocktailPreferences(),
            "Empfiehl mir bitte etwas Cremiges",
            cocktails,
        )

        self.assertEqual(preferences.liked_flavors, ["cremig"])

    def test_recommendation_explanation_resolves_pronoun_without_excluding_flavor(self):
        pina_colada = {
            **COCKTAILS[0],
            "name": "Pina Colada",
            "geschmack": ["süß", "fruchtig", "cremig"],
        }

        class PinaRepository:
            def list_all(self):
                return COCKTAILS + [pina_colada]

        history = [
            chat_service.ChatMessage(
                role="user", content="Aber die Pina Colada ist doch auch cremig, oder nicht?"
            ),
            chat_service.ChatMessage(
                role="assistant", content="Ja, Pina Colada ist laut unserer Karte cremig."
            ),
        ]
        response = asyncio.run(
            chat_service.build_chat_response(
                "Wieso schlägst du ihn uns dann nicht vor bei cremig?",
                history=history,
                repository=PinaRepository(),
                session_id="recommendation-explanation",
                conversations=ConversationService(),
            )
        )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertIn("Pina Colada passt ebenfalls zu cremig", response["answer"])
        self.assertIn("höchstens drei", response["answer"])
        self.assertNotIn("ohne cremig", response["answer"])
        self.assertEqual(response["preferences"]["disliked_flavors"], [])

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

    def test_spirit_preference_is_not_duplicated_as_ingredient(self):
        preferences = chat_service.local_update_preferences(
            chat_service.CocktailPreferences(),
            "Ich mag Rum",
            COCKTAILS,
        )

        self.assertEqual(preferences.spirits, ["Rum"])
        self.assertEqual(preferences.liked_ingredients, [])

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

    def test_random_response_without_preferences_uses_all_cocktails(self):
        with patch.object(chat_service.random, "choice", side_effect=lambda items: items[0]):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ueberrasch mich",
                    repository=FakeRepository(),
                    conversations=ConversationService(),
                )
            )

        self.assertEqual(response["type"], "random")
        self.assertEqual(response["intent"], "random")
        self.assertEqual(response["selected_cocktail"]["name"], "Caribbean Dream")
        self.assertEqual(len(response["roulette_cocktails"]), len(COCKTAILS))
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Caribbean Dream"])

    def test_random_response_respects_rum_and_coconut_exclusion(self):
        with patch.object(chat_service.random, "choice", side_effect=lambda items: items[0]):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Ich mag Rum und keinen Kokos. Ueberrasch mich.",
                    repository=FakeRepository(),
                    conversations=ConversationService(),
                )
            )

        self.assertEqual(response["type"], "random")
        self.assertEqual(response["preferences"]["spirits"], ["Rum"])
        self.assertIn("Kokossirup", response["preferences"]["disliked_ingredients"])
        self.assertEqual(response["selected_cocktail"]["name"], "Caribbean Dream")
        self.assertTrue(
            all("Kokossirup" not in cocktail["zutaten"] for cocktail in response["roulette_cocktails"])
        )

    def test_random_response_avoids_immediate_repeat_when_possible(self):
        conversations = ConversationService()
        conversations.update_last_random_cocktail("again", "Caribbean Dream")

        with patch.object(chat_service.random, "choice", side_effect=lambda items: items[0]):
            response = asyncio.run(
                chat_service.build_chat_response(
                    "Nochmal!",
                    repository=FakeRepository(),
                    session_id="again",
                    conversations=conversations,
                )
            )

        self.assertEqual(response["type"], "random")
        self.assertNotEqual(response["selected_cocktail"]["name"], "Caribbean Dream")

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

    def test_generated_recommendation_ignores_free_llm_answer(self):
        async def fake_query_ollama(*args, **kwargs):
            return json.dumps({
                "answer": "Ich empfehle den frei erfundenen Galaxy Cocktail.",
                "cocktail_names": ["Gin Sour"],
            })

        with patch.object(chat_service, "query_ollama", fake_query_ollama):
            response = asyncio.run(
                chat_service.generate_answer("Empfiehl mir etwas", [COCKTAILS[2]])
            )

        self.assertNotIn("Galaxy Cocktail", response["answer"])
        self.assertIn("Gin Sour", response["answer"])
        self.assertEqual([item["name"] for item in response["cocktails"]], ["Gin Sour"])

    def test_normalize_search_criteria_infers_fruity_taste_from_message(self):
        criteria = CocktailSearchCriteria()

        normalized = chat_service.normalize_search_criteria(
            criteria,
            "Ich moechte einen fruchtigen Cocktail",
            COCKTAILS,
        )

        self.assertEqual(normalized.geschmack, "fruchtig")
