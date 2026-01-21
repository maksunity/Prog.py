import sys
import os


def main():
    print("КОМАНДЫ ДЛЯ ЗАПУСКА:")
    print()
    print("1. Сбор общих новостей (лимит 100):")
    print('   python run_spider.py --spider news --keywords "" --days 7 --output news.json')
    print()
    print("2. Сбор новостей по тегам (технологии, наука):")
    print('   python run_spider.py --spider news --keywords "технологии,наука" --days 7 --output news_tech.json')
    print()
    print("3. Сбор новостей ПНИПУ (лимит 100):")
    print('   python run_spider.py --spider news --keywords "политех,пнипу" --days 30 --output pstu.json')
    print()
    print("4. Сбор новостей по ключевым словам (политика, экономика):")
    print('   python run_spider.py --spider news --keywords "политика,экономика" --days 7 --output news.json')
    print()
    print("=" * 70)
    print()
    print("Источники новостей:")
    print("  - Lenta.ru")
    print("  - RBC.ru")
    print("  - TASS.ru")
    print("  - RIA.ru")
    print("  - PSTU.ru (ПНИПУ)")
    print()
    print("Особенности:")
    print("  - Автоматическое извлечение заголовков и дат")
    print("  - Фильтрация по ключевым словам")
    print("  - Обработка ошибок и повторные попытки")
    print("  - Экспорт в JSON и CSV")
    print("  - Лимит: 100 записей за запуск")
    print()
    print("=" * 70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            print("\nЗапускаю тест news spider...")
            os.system('python run_spider.py --spider news --keywords "" --days 7 --output test_news.json')
        elif sys.argv[1] == 'pstu':
            print("\nЗапускаю парсинг новостей ПНИПУ...")
            os.system('python run_spider.py --spider news --keywords "политех,пнипу" --days 30 --output pstu.json')
        elif sys.argv[1] == 'tech':
            print("\nЗапускаю парсинг новостей о технологиях...")
            os.system('python run_spider.py --spider news --keywords "технологии,наука" --days 7 --output news_tech.json')
        elif sys.argv[1] == 'run':
            print("\nЗапускаю spider...")
            os.system('python run_spider.py ' + ' '.join(sys.argv[2:]))
    else:
        print("\nБыстрый запуск:")
        print("  python main.py test     - общие новости")
        print("  python main.py pstu     - новости ПНИПУ")
        print("  python main.py tech     - новости о технологиях")


if __name__ == '__main__':
    main()

