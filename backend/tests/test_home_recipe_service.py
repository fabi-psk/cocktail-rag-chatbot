import asyncio
import unittest
from unittest.mock import patch

from services import home_recipe_service


IBA_HTML = """
<!doctype html>
<html>
  <head><title>Mojito &#8211; IBA</title></head>
  <body>
    <h4>Ingredients</h4>
    <div><ul>
      <li>45 ml White Cuban Ron</li>
      <li>20 ml Fresh Lime Juice</li>
      <li>6 pcs Mint Sprigs</li>
      <li>2 tsp White Cane Sugar</li>
      <li>Soda Water</li>
    </ul></div>
    <h4>Method</h4>
    <div>
      <p>Mix mint sprigs with sugar and lime juice.</p>
      <p>Pour the rum and top with soda water.</p>
    </div>
    <h4>Garnish</h4>
    <p>Garnish with mint and lime.</p>
    <footer><p>Are you over 18 years of age?</p></footer>
  </body>
</html>
"""


class HomeRecipeServiceTest(unittest.TestCase):
    def test_recipe_slug_is_safe_and_predictable(self):
        self.assertEqual(home_recipe_service.recipe_slug("  Piña Colada  "), "pina-colada")
        self.assertEqual(home_recipe_service.recipe_slug("Mai-Tai"), "mai-tai")

    def test_parse_iba_recipe_extracts_recipe_sections(self):
        recipe = home_recipe_service.parse_iba_recipe(
            IBA_HTML, "https://iba-world.com/iba-cocktail/mojito/"
        )

        self.assertEqual(recipe["name"], "Mojito")
        self.assertEqual(recipe["ingredients"][0], "45 ml White Cuban Ron")
        self.assertEqual(len(recipe["ingredients"]), 5)
        self.assertEqual(recipe["instructions"][1], "Pour the rum and top with soda water.")
        self.assertEqual(recipe["garnish"], ["Garnish with mint and lime."])

    def test_home_response_uses_extracted_web_recipe(self):
        recipe = home_recipe_service.parse_iba_recipe(
            IBA_HTML, "https://iba-world.com/iba-cocktail/mojito/"
        )

        async def cocktail_name(*args, **kwargs):
            return "Mojito"

        async def fetched_recipe(*args, **kwargs):
            return recipe

        with patch.object(home_recipe_service, "extract_requested_cocktail", cocktail_name):
            with patch.object(home_recipe_service, "fetch_iba_recipe", fetched_recipe):
                response = asyncio.run(
                    home_recipe_service.build_home_recipe_response(
                        "Wie mache ich einen Mojito?"
                    )
                )

        self.assertEqual(response["intent"], "catalog_query")
        self.assertEqual(response["web_recipes"][0]["name"], "Mojito")
        self.assertIn("45 ml White Cuban Ron", response["answer"])
        self.assertIn("https://iba-world.com/iba-cocktail/mojito/", response["answer"])

    def test_missing_source_recipe_does_not_invent_ingredients(self):
        async def cocktail_name(*args, **kwargs):
            return "Fantasy Drink"

        async def missing_recipe(*args, **kwargs):
            return None

        with patch.object(home_recipe_service, "extract_requested_cocktail", cocktail_name):
            with patch.object(home_recipe_service, "fetch_iba_recipe", missing_recipe):
                response = asyncio.run(
                    home_recipe_service.build_home_recipe_response("Fantasy Drink Rezept")
                )

        self.assertEqual(response["web_recipes"], [])
        self.assertIn("erfinde deshalb keine Zutaten", response["answer"])


if __name__ == "__main__":
    unittest.main()
