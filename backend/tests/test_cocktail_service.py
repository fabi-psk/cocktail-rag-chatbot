import unittest

from models.cocktail import CocktailSearchCriteria
from services.cocktail_service import search_cocktails


COCKTAILS = [
    {
        "name": "Zombie",
        "preis": 9.5,
        "spirituose": ["Heller Rum", "Dunkler Rum"],
        "geschmack": ["fruchtig", "sauer"],
        "staerke": "hoch",
        "zutaten": ["Heller Rum", "Dunkler Rum", "Ananassaft", "Orangensaft"],
        "beschreibung": "Starker Rum-Cocktail.",
    },
    {
        "name": "Bahama Mama",
        "preis": 9.0,
        "spirituose": ["Dunkler Rum", "Malibu"],
        "geschmack": ["fruchtig", "tropisch"],
        "staerke": "hoch",
        "zutaten": ["Dunkler Rum", "Kokossirup", "Ananassaft"],
        "beschreibung": "Stark und tropisch mit Kokos.",
    },
    {
        "name": "Mojito",
        "preis": 7.5,
        "spirituose": ["Rum"],
        "geschmack": ["frisch", "minzig"],
        "staerke": "mittel",
        "zutaten": ["Rum", "Limette", "Minze", "Soda"],
        "beschreibung": "Frischer kubanischer Cocktail.",
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


class CocktailServiceTest(unittest.TestCase):
    def test_search_rum_and_strong_maps_stark_to_hoch(self):
        criteria = CocktailSearchCriteria(spirituose="Rum", staerke="stark")

        result = search_cocktails(criteria, COCKTAILS)

        self.assertEqual([cocktail["name"] for cocktail in result], ["Zombie", "Bahama Mama"])

    def test_search_excludes_ingredient(self):
        criteria = CocktailSearchCriteria(spirituose="Rum", staerke="stark", ausschluesse=["Kokos"])

        result = search_cocktails(criteria, COCKTAILS)

        self.assertEqual([cocktail["name"] for cocktail in result], ["Zombie"])

    def test_search_combines_multiple_criteria(self):
        criteria = CocktailSearchCriteria(spirituose="Gin", geschmack="sauer")

        result = search_cocktails(criteria, COCKTAILS)

        self.assertEqual([cocktail["name"] for cocktail in result], ["Gin Sour"])

    def test_search_no_matches(self):
        criteria = CocktailSearchCriteria(spirituose="Tequila", geschmack="cremig", staerke="stark")

        result = search_cocktails(criteria, COCKTAILS)

        self.assertEqual(result, [])
