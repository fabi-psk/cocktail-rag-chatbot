import json

with open("../data/cocktails.json", "r", encoding="utf-8") as file:
    cocktails = json.load(file)

frage = input("Welche Spirituose suchst du? ")

print("\nGefundene Cocktails:")

for cocktail in cocktails:
    if frage in cocktail["spirituose"]:
        print("-", cocktail["name"])
