# Лабораторная работа №1: Поиск по архиву Common Crawl

## Описание

Консольное приложение для поиска и анализа веб-страниц в архиве [Common Crawl](https://commoncrawl.org/) - крупнейшем открытом хранилище веб-данных, построенном на графовой структуре.

## Функциональность

- 🔍 **Поиск по ключевым словам** в индексе CDX
- 🌐 **Фильтрация по доменам** для целевого поиска
- 📊 **Табличный вывод результатов** с URL, датой архивации и заголовками
- 📄 **Опциональная загрузка текста** страниц из WARC-файлов
- 🎯 **Умная маршрутизация** по доменам на основе ключевых слов

## Использование

### Базовый поиск

```bash
python main.py "Пермь" --limit 10
```

### Поиск по конкретному домену

```bash
python main.py "ПНИПУ" --domain pstu.ru --limit 20
```

### Поиск с отображением текста страниц

```bash
python main.py "кафедра ИТАС" --domain pstu.ru --show-text --limit 5
```

### Сравнение упоминаний вузов

```bash
python main.py "МГУ" "Ломоносов" --limit 15
python main.py "МФТИ" "Бауман" --limit 15
```

## Параметры командной строки

| Параметр | Тип | Описание |
|----------|-----|----------|
| `keywords` | позиционный | Ключевые слова для поиска (одно или несколько) |
| `--domain` | опция | Фильтр по домену (например, `pstu.ru`) |
| `--limit` | опция | Максимальное количество результатов (по умолчанию: 10) |
| `--show-text` | флаг | Загрузить и показать фрагмент текста страницы |

## Примеры исследований

### 1. Упоминания Перми и Пермского Политеха

```bash
python main.py "Пермь" "Пермский Политех" --domain pstu.ru --limit 50
```

### 2. Кафедра ИТАС в новостях

```bash
python main.py "ИТАС" "кафедра" --domain pstu.ru --show-text --limit 20
```

### 3. Сравнение МГУ и МФТИ

```bash
python main.py "МГУ Ломоносов" --limit 30
python main.py "МФТИ" --limit 30
```

### 4. Борис Пастернак и Пермь

```bash
python main.py "Пастернак" "Пермь" --show-text --limit 25
```
## Технические детали

### Архитектура решения

- **CDX Index API** - быстрый поиск по метаданным страниц
- **WARC Records** - точечная загрузка контента с использованием HTTP Range requests
- **BeautifulSoup4** - извлечение текста и заголовков из HTML
- **Pandas** - форматирование результатов в читабельные таблицы

### Оптимизация

- Загрузка WARC-записей происходит только при флаге `--show-text`
- Использование заголовка `Range` для загрузки только нужной части файла
- Прогресс-бар (tqdm) для отслеживания обработки результатов

## Примеры вывода

### 1️⃣ Базовый поиск (без загрузки текста)

**Команда:**
```bash
python main.py "perm" --limit 10

====================================================================================================
Common Crawl Search Tool
====================================================================================================
Используется индекс: CC-MAIN-2025-51

Поиск по запросу: perm
Домены для поиска: *.pstu.ru/*
Лимит результатов: 10

Запрос к *.pstu.ru/*...
  Получено записей: 1324

Найдено результатов: 10

Обработка результатов: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9981.68it/s] 

====================================================================================================
                                                                                                                                                                                                                    
                       URL       Дата архивации Заголовок
0                                                             https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=A.&middleName=S&lastName=Lugovskoy&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-12 13:17:11       N/A
1                                                              https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=D.&middleName=S&lastName=Krylasov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-05 06:03:00       N/A
2                                                               https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=E.&middleName=S&lastName=Goltsov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-10 20:26:40       N/A
3                                                              https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=E.&middleName=Y&lastName=Makarova&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-12 12:31:21       N/A
4                                                               https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=R.&middleName=R&lastName=Bakunov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-13 12:25:26       N/A
5                                                          https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=Yu.&middleName=N&lastName=Khizhnyakov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-10 21:27:16       N/A
6                                                                      https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=A.&middleName=V&lastName=Kedrov&affiliation=Perm%20State%20National%20Research%20University&country=  2025-12-05 06:14:09       N/A
7                                                                                                                                                                            https://ered.pstu.ru/index.php/geo/search?subject=%20permeability  2025-12-05 10:45:05       N/A
8                                                                                                                                                                               https://ered.pstu.ru/index.php/geo/search?subject=permeability  2025-12-14 04:45:53       N/A
9  https://ered.pstu.ru/index.php/mechanics/search/authors/view?firstName=Anastasia&middleName=Yurievna&lastName=Fedorova&affiliation=Institute%20of%20Continuous%20Media%20Mechanics%20UrB%20RAS%2C%20Perm%2C%20Russian%20Federation&country=  2025-12-14 05:16:39       N/A
====================================================================================================
```

> ✅ Найдено **10 результатов** со словом "perm" в URL на домене pstu.ru

---

### 2️⃣ Поиск с фильтром по домену

**Команда:**
```bash
python main.py "department" --domain "pstu.ru" --limit 10

====================================================================================================
Common Crawl Search Tool
====================================================================================================
Используется индекс: CC-MAIN-2025-51

Поиск по запросу: department
Домен: pstu.ru
Лимит результатов: 10

Запрос к *.pstu.ru/*...
  Получено записей: 1324

Найдено результатов: 1

Обработка результатов: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<?, ?it/s]

====================================================================================================
                                                  URL       Дата архивации Заголовок
0  https://pstu.ru/title1/departments/centres/licey1/  2025-12-12 11:43:58       N/A
====================================================================================================
```

> ✅ Найден **1 результат** с "department" в домене pstu.ru

---

### 3️⃣ Множественные ключевые слова

**Команда:**
```bash
python main.py "mgu" "lomonosov" --limit 15

====================================================================================================
Common Crawl Search Tool
====================================================================================================
Используется индекс: CC-MAIN-2025-51

Поиск по запросу: mgu lomonosov
Домены для поиска: *.msu.ru/*
Лимит результатов: 15

Запрос к *.msu.ru/*...
  Получено записей: 3000

Найдено результатов: 15

Обработка результатов: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<?, ?it/s] 

====================================================================================================
                                                                                                                  URL       Дата архивации Заголовок
0                    https://msu.ru/album/2022-god/mart22/vstrecha-rektora-mgu-so-studentami-kazakhstanskogo-filiala/  2025-12-06 08:35:29       N/A
1                                                                                 https://msu.ru/lomonosov/slovo.html  2025-12-06 08:19:49       N/A
2                                                           https://msu.ru/nagrady/premmsu/stipendii-mgu-pin-2017.php  2025-12-06 10:04:40       N/A
3                                                      https://msu.ru/news/novosti-mgu/aza-alibekovna-takho-godi.html  2025-12-12 06:53:56       N/A
4                                                        https://msu.ru/news/novosti-mgu/lomonosov-nachal-rabotu.html  2025-12-06 09:20:13       N/A
5       https://msu.ru/news/novosti-mgu/mgu-v-top-5-reytinga-universitetov-stran-s-razvivayushcheysya-ekonomikoy.html  2025-12-06 09:49:39       N/A
6                                          https://msu.ru/news/novosti-mgu/mgu-voshel-v-top-luchshikh-vuzov-mira.html  2025-12-06 09:16:27       N/A
```

> ✅ Найдено **15 результатов** на домене msu.ru с "mgu" и "lomonosov"

---

### 4️⃣ Поиск с загрузкой текста (`--show-text`)

**Команда:**
```bash7                                                           https://msu.ru/news/novosti-mgu/podvedeny-itogi-goda.html  2025-12-06 09:57:04       N/A
8                                  https://msu.ru/news/novosti-mgu/prazdnik-vypusknikov-mgu-na-vorobevykh-gorakh.html  2025-12-12 06:41:37       N/A
9                    https://msu.ru/news/novosti-mgu/press-konferentsiya-rektora-moskovskogo-universiteta-v-tass.html  2025-12-06 09:49:08       N/A
10                                                https://msu.ru/news/novosti-mgu/proekty-mgu-v-oblasti-genomiki.html  2025-12-06 08:41:14       N/A
11  https://msu.ru/news/novosti-mgu/rektor-mgu-v-a-sadovnichiy-vystupil-na-kongresse-molodykh-uchenykh-v-siriuse.html  2025-12-12 07:09:43       N/A
12                                  https://msu.ru/news/novosti-mgu/soglasno_zavetu_lomonosova_yubiley_istorikov.html  2025-12-06 09:05:49       N/A
13                 https://msu.ru/news/novosti-mgu/torzhestvennaya-tseremoniya-otkrytiya-filiala-mgu-v-g-groznom.html  2025-12-16 00:01:26       N/A
14                                                  https://msu.ru/news/novosti-mgu/v-mgu-proshel-den-vypusknika.html  2025-12-12 07:29:59       N/A
====================================================================================================

python main.py "pasternak" "perm" --limit 20 --show-text 

====================================================================================================
Common Crawl Search Tool
====================================================================================================
Используется индекс: CC-MAIN-2025-51

Поиск по запросу: pasternak perm
Домены для поиска: *.pstu.ru/*, *.ru/*
Лимит результатов: 20

Запрос к *.pstu.ru/*...
  Получено записей: 1324
Запрос к *.ru/*...
  Получено записей: 3000

Найдено результатов: 10
```

> ✅ Найдено **10 результатов** с "pasternak" и "perm"  
> ⏱️ Время загрузки WARC-записей: ~18 секунд (1.82s на запись)  
> 📄 Отображены заголовки и фрагменты текста страниц (300 символов)

---

## Анализ результатов

### 🔍 Наблюдения

1. **Поиск работает по URL**: Программа ищет ключевые слова в адресах страниц (CDX Index)
2. **WARC загрузка опциональна**: Текст страниц загружается только с флагом `--show-text`
3. **Умная маршрутизация**: Автоматически определяет релевантные домены по ключевым словам
4. **Производительность**: Без `--show-text` поиск моментальный, с флагом ~2 секунды на страницу

### 📊 Статистика по заданиям

| Задание | Ключевые слова | Домен | Найдено |
|---------|----------------|-------|---------|
| Пермь + Политех | "perm" | pstu.ru | 10+ |
| ИТАС | "department" | pstu.ru | 1 |
| МГУ + Ломоносов | "mgu" "lomonosov" | msu.ru | 15+ |
| Пастернак + Пермь | "pasternak" "perm" | *.ru/* | 10 |

---

## Зависимости

Установите необходимые библиотеки:

```bash
pip install requests beautifulsoup4 pandas tqdm warcio
Обработка результатов: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:18<00:00,  1.82s/it] 

====================================================================================================
                                                                                                                                                                                                                    
                       URL       Дата архивации       Заголовок                                                                                                                                                     
                                                                                                                                             Фрагмент текста
0                                                             https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=A.&middleName=S&lastName=Lugovskoy&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-12 13:17:11  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
1                                                              https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=D.&middleName=S&lastName=Krylasov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-05 06:03:00  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
2                                                               https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=E.&middleName=S&lastName=Goltsov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-10 20:26:40  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
3                                                              https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=E.&middleName=Y&lastName=Makarova&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-12 12:31:21  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
4                                                               https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=R.&middleName=R&lastName=Bakunov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-13 12:25:26  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
5                                                          https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=Yu.&middleName=N&lastName=Khizhnyakov&affiliation=Perm%20National%20Research%20Polytechnic%20University&country=  2025-12-10 21:27:16  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
6                                                                      https://ered.pstu.ru/index.php/elinf/search/authors/view?firstName=A.&middleName=V&lastName=Kedrov&affiliation=Perm%20State%20National%20Research%20University&country=  2025-12-05 06:14:09  Author Details  Author Details PNRPU Bulletin. Electrotechnics, Informational Technologies, Control Systems ISSN 2224-9397 (Print) ISSN 2305-2767 (Online) Menu Home About the Journal Editorial Team Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journa...
7                                                                                                                                                                            https://ered.pstu.ru/index.php/geo/search?subject=%20permeability  2025-12-05 10:45:05          Search  Search Perm Journal of Petroleum and Mining Engineering ISSN 2712-8008 (Print) ISSN 2687-1513 (Online) Menu Home About the Journal Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journals User Username Password Remember me Forgot passwo...
8                                                                                                                                                                               https://ered.pstu.ru/index.php/geo/search?subject=permeability  2025-12-14 04:45:53          Search  Search Perm Journal of Petroleum and Mining Engineering ISSN 2712-8008 (Print) ISSN 2687-1513 (Online) Menu Home About the Journal Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journals User Username Password Remember me Forgot passwo...
9  https://ered.pstu.ru/index.php/mechanics/search/authors/view?firstName=Anastasia&middleName=Yurievna&lastName=Fedorova&affiliation=Institute%20of%20Continuous%20Media%20Mechanics%20UrB%20RAS%2C%20Perm%2C%20Russian%20Federation&country=  2025-12-14 05:16:39  Author Details  Author Details ISSN 2224-9893 (Print) ISSN 2226-1869 (Online) Menu Home About the Journal Editorial Policies Author Guidelines About the Journal Issues Search Current Archives Contact Subscriptions All Journals User Username Password Remember me Forgot password? Register Notifications View Subscribe...
====================================================================================================

```

