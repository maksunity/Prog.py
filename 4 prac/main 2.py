import json
from collections import Counter

with open('countries-data.json', 'r', encoding='utf-8') as file:
    countries = json.load(file)

sorted_by_name = sorted(countries, key=lambda x: x['name'])

sorted_by_capital = sorted(countries, key=lambda x: x['capital'])

sorted_by_population = sorted(countries, key=lambda x: x['population'], reverse=True)

all_languages = [language for country in countries for language in country['languages']]
language_counts = Counter(all_languages)
most_common_languages = language_counts.most_common(10)

language_usage = {
    lang: [country['name'] for country in countries if lang in country['languages']]
    for lang, _ in most_common_languages
}

most_populous_countries = sorted(countries, key=lambda x: x['population'], reverse=True)[:10]

print("\nSorted by name:")
for country in sorted_by_name:
    print(country['name'])

print("\nSorted by capital:")
for country in sorted_by_capital:
    print(country['capital'])

print("\nSorted by population:")
for country in sorted_by_population:
    print(f"{country['name']}: {country['population']}")

print("\nMost common languages:")
for lang, count in most_common_languages:
    print(f"{lang}: {count} users")
    print(f"Countries: {', '.join(language_usage[lang])}")

print("\nMost populous countries:")
for country in most_populous_countries:
    print(f"{country['name']}: {country['population']}")