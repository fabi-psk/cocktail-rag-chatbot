import asyncio
import unittest
from unittest.mock import patch

from services import home_recipe_service


INDEX_HTML = """
<html><body>
  <a href="/cocktail-rezepte/new">Rezept einsenden</a>
  <a href="/cocktail-rezepte/254-mai-tai#autoplay">Mai Tai</a>
  <a href="/cocktail-rezepte/140-mojito#autoplay">Mojito</a>
  <a href="/cocktail-rezepte/16-manhattan">Manhattan</a>
</body></html>
"""


COCKTAIL_DB_HTML = """
<!doctype html>
<html><body>
  <h1><span itemprop="name">Mojito</span> Rezept</h1>
  <section>
    <h2>Zutaten</h2>
    <ul class="recipe-ingredients">
      <li itemprop="ingredients"><span>2 TL</span> <a>Rohrzucker</a></li>
      <li itemprop="ingredients"><span>10</span> <a>Minzeblätter</a></li>
      <li itemprop="ingredients"><span>6 cl</span> <a>weißer Rum</a></li>
      <li itemprop="ingredients"><span>4 cl</span> <a>Sodawasser</a></li>
      <li itemprop="ingredients"><span>1</span> <a>Limette(n)</a></li>
    </ul>
  </section>
  <section>
    <h2>Zubereitung</h2>
    <p>Die Limette in vier Stücke teilen. Rohrzucker und Minzeblätter hinzufügen. Mit Eis, Rum und Sodawasser auffüllen und gut umrühren.</p>
  </section>
  <section><h2>Bewertungen</h2><p>Dieser Text gehört nicht zum Rezept.</p></section>
</body></html>
"""


class HomeRecipeServiceTest(unittest.TestCase):
    def test_recipe_index_key_is_safe_and_predictable(self):
        self.assertEqual(home_recipe_service.recipe_index_key("  Piña Colada  "), "p")
        self.assertEqual(home_recipe_service.recipe_index_key("Mai-Tai"), "m")
        self.assertEqual(home_recipe_service.recipe_index_key("42 Cocktail"), "0-9")
        self.assertIsNone(home_recipe_service.recipe_index_key("---"))

    def test_recipe_path_is_resolved_without_query_parameters(self):
        self.assertEqual(
            home_recipe_service.find_recipe_path(INDEX_HTML, "Mojito"),
            "/cocktail-rezepte/140-mojito",
        )
        self.assertEqual(
            home_recipe_service.find_recipe_path(INDEX_HTML, "Mai-Tai"),
            "/cocktail-rezepte/254-mai-tai",
        )
        self.assertIsNone(
            home_recipe_service.find_recipe_path(INDEX_HTML, "Fantasy Drink")
        )

    def test_parse_cocktail_database_recipe_extracts_only_recipe_sections(self):
        recipe = home_recipe_service.parse_cocktail_database_recipe(
            COCKTAIL_DB_HTML,
            "https://www.cocktaildatenbank.de/cocktail-rezepte/140-mojito",
        )

        self.assertEqual(recipe["name"], "Mojito")
        self.assertEqual(recipe["ingredients"][0], "2 TL Rohrzucker")
        self.assertEqual(len(recipe["ingredients"]), 5)
        self.assertEqual(len(recipe["instructions"]), 3)
        self.assertNotIn("Bewertungen", " ".join(recipe["instructions"]))
        self.assertEqual(recipe["source_name"], "Cocktaildatenbank.de")

    def test_home_response_uses_german_cocktail_database_recipe(self):
        recipe = home_recipe_service.parse_cocktail_database_recipe(
            COCKTAIL_DB_HTML,
            "https://www.cocktaildatenbank.de/cocktail-rezepte/140-mojito",
        )

        async def cocktail_name(*args, **kwargs):
            return "Mojito"

        async def fetched_recipe(*args, **kwargs):
            return recipe

        async def rewritten_recipe(source_recipe):
            return {
                **source_recipe,
                "instructions": [
                    "Limette vierteln.",
                    "Rohrzucker und Minzblätter dazugeben.",
                    "Eis und Rum hinzufügen, mit Sodawasser auffüllen und umrühren.",
                ],
            }

        with (
            patch.object(home_recipe_service, "extract_requested_cocktail", cocktail_name),
            patch.object(
                home_recipe_service,
                "fetch_cocktail_database_recipe",
                fetched_recipe,
            ),
            patch.object(home_recipe_service, "rewrite_recipe_with_llm", rewritten_recipe),
        ):
            response = asyncio.run(
                home_recipe_service.build_home_recipe_response(
                    "Wie mache ich einen Mojito?"
                )
            )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual(response["web_recipes"][0]["name"], "Mojito")
        self.assertIn("2 TL Rohrzucker", response["answer"])
        self.assertIn("Limette vierteln", response["answer"])
        self.assertIn("cocktaildatenbank.de/cocktail-rezepte/140-mojito", response["answer"])

    def test_llm_rewrite_keeps_recipe_structure_and_source(self):
        recipe = home_recipe_service.parse_cocktail_database_recipe(
            COCKTAIL_DB_HTML,
            "https://www.cocktaildatenbank.de/cocktail-rezepte/140-mojito",
        )

        async def fake_query(*args, **kwargs):
            return """{
                "ingredients": [
                    "2 TL Rohrzucker",
                    "10 Minzeblätter",
                    "6 cl weißer Rum",
                    "4 cl Sodawasser",
                    "1 Limette"
                ],
                "instructions": [
                    "Limette vierteln.",
                    "Zucker und Minze hinzufügen.",
                    "Mit Eis, Rum und Sodawasser auffüllen und umrühren."
                ],
                "garnish": []
            }"""

        with patch.object(home_recipe_service, "query_ollama", fake_query):
            rewritten = asyncio.run(home_recipe_service.rewrite_recipe_with_llm(recipe))

        self.assertEqual(rewritten["name"], "Mojito")
        self.assertEqual(len(rewritten["ingredients"]), len(recipe["ingredients"]))
        self.assertEqual(rewritten["source_url"], recipe["source_url"])
        self.assertIn("auffüllen", rewritten["instructions"][2])

    def test_unrelated_llm_step_falls_back_to_german_source(self):
        source = "Als Dekoration einen Pfefferminzzweig ins Glas geben."
        self.assertFalse(
            home_recipe_service.rewritten_step_is_grounded(source, "Rum eingießen.")
        )
        self.assertTrue(
            home_recipe_service.rewritten_step_is_grounded(
                source, "Mit einem Pfefferminzzweig dekorieren."
            )
        )

    def test_missing_source_recipe_does_not_invent_ingredients(self):
        async def cocktail_name(*args, **kwargs):
            return "Fantasy Drink"

        async def missing_recipe(*args, **kwargs):
            return None

        with (
            patch.object(home_recipe_service, "extract_requested_cocktail", cocktail_name),
            patch.object(
                home_recipe_service,
                "fetch_cocktail_database_recipe",
                missing_recipe,
            ),
        ):
            response = asyncio.run(
                home_recipe_service.build_home_recipe_response("Fantasy Drink Rezept")
            )

        self.assertEqual(response["web_recipes"], [])
        self.assertIn("erfinde deshalb keine Zutaten", response["answer"])


if __name__ == "__main__":
    unittest.main()
