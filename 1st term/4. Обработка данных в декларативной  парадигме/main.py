import json
from functools import reduce

with open('countries.json', 'r', encoding='utf-8') as file:
    countries = json.load(file)

uppercase_countries = list(map(str.upper, countries))
print("Uppercase countries:", uppercase_countries, "\n")

land_countries = list(filter(lambda country: 'land' in country.lower(), countries))
print("Countries containing 'land':", land_countries, "\n")

six_letter_countries = list(filter(lambda country: len(country) == 6, countries))
print("Countries with six letters:", six_letter_countries, "\n")

six_or_more_letter_countries = list(filter(lambda country: len(country) >= 6, countries))
print("Countries with six or more letters:", six_or_more_letter_countries, "\n")

non_e_countries = list(filter(lambda country: not country.startswith('E'), countries))
print("Countries not starting with 'E':", non_e_countries, "\n")

nordic_countries = ["Finland", "Sweden", "Denmark", "Norway", "Iceland"]
nordic_sentence = reduce(lambda acc, country: f"{acc}, {country}" if acc else country,
    nordic_countries, "") + " являются странами Северной Европы."
print("Nordic countries sentence:", nordic_sentence, "\n")

filtered_and_uppercase = list(map(str.upper, filter(lambda country: 'land' in country.lower(), countries)))
print("Filtered and uppercase countries containing 'land':", filtered_and_uppercase, "\n")

# замыкание
def categorize_countries(pattern):
    def filter_countries(countries):
        return list(filter(lambda country: pattern in country.lower(), countries))
    return filter_countries

land_categorizer = categorize_countries('land')
print("Categorized countries (land):", land_categorizer(countries), "\n")

# каррирование
def curry_categorize(pattern):
    return lambda countries: list(filter(lambda country: pattern in country.lower(), countries))

ia_categorizer = curry_categorize('ia')
print("Categorized countries (ia):", ia_categorizer(countries), "\n")
