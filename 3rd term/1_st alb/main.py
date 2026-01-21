import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Поиск по архиву Common Crawl'
    )
    parser.add_argument(
        'keywords',
        nargs='+',
        help='Ключевые слова для поиска'
    )
    parser.add_argument(
        '--domain',
        help='Ограничить поиск определенным доменом'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Максимальное количество результатов (по умолчанию: 10)'
    )
    parser.add_argument(
        '--show-text',
        action='store_true',
        help='Показать фрагмент текста найденной страницы'
    )
    return parser.parse_args()


def get_latest_index():
    try:
        response = requests.get('https://index.commoncrawl.org/collinfo.json', timeout=10)
        response.raise_for_status()
        indexes = response.json()
        if indexes:
            latest = indexes[0]['id']
            print(f"Используется индекс: {latest}")
            return latest
    except Exception as e:
        print(f"Не удалось получить список индексов: {e}")
    
    return 'CC-MAIN-2025-51'


def search_cdx_index(keywords, domain=None, limit=10):
    results = []
    
    index_name = get_latest_index()
    cdx_api_url = f'https://index.commoncrawl.org/{index_name}-index'
    
    search_query = ' '.join(keywords)
    
    search_domains = []
    
    if domain:
        search_domains = [f'*.{domain}/*']
    else:
        keywords_lower = [kw.lower() for kw in keywords]
        
        for kw in keywords_lower:
            if 'perm' in kw or 'пермь' in kw or 'пермск' in kw:
                search_domains.append('*.pstu.ru/*')
                break
        
        if 'itac' in ' '.join(keywords_lower) or 'итас' in ' '.join(keywords_lower):
            if '*.pstu.ru/*' not in search_domains:
                search_domains.append('*.pstu.ru/*')
        
        if 'mgu' in ' '.join(keywords_lower) or 'мгу' in ' '.join(keywords_lower) or 'lomonosov' in ' '.join(keywords_lower):
            search_domains.append('*.msu.ru/*')
        
        if 'mfti' in ' '.join(keywords_lower) or 'мфти' in ' '.join(keywords_lower) or 'bauman' in ' '.join(keywords_lower):
            search_domains.append('*.mipt.ru/*')
            search_domains.append('*.bmstu.ru/*')
        
        if 'pasternak' in ' '.join(keywords_lower) or 'пастернак' in ' '.join(keywords_lower):
            search_domains.append('*.ru/*')
        
        if not search_domains:
            search_domains.append('*.ru/*')
    
    print(f"\nПоиск по запросу: {search_query}")
    if domain:
        print(f"Домен: {domain}")
    else:
        print(f"Домены для поиска: {', '.join(search_domains)}")
    print(f"Лимит результатов: {limit}\n")
    
    all_results = []
    
    for url_pattern in search_domains:
        params = {
            'url': url_pattern,
            'output': 'json',
            'limit': 3000
        }
        
        try:
            print(f"Запрос к {url_pattern}...")
            response = requests.get(cdx_api_url, params=params, timeout=90)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            print(f"  Получено записей: {len(lines)}")
            
            search_keywords_lower = [kw.lower() for kw in keywords]
            
            for line in lines:
                if not line:
                    continue
                try:
                    import json
                    data = json.loads(line)
                    url = data.get('url', '')
                    url_lower = url.lower()
                    status = data.get('status', '')
                    
                    match = False
                    for keyword in search_keywords_lower:
                        if keyword in url_lower:
                            match = True
                            break
                    
                    if match and status == '200':
                        mime = data.get('mime', '')
                        if 'html' in mime.lower():
                            all_results.append({
                                'url': url,
                                'timestamp': data.get('timestamp', ''),
                                'filename': data.get('filename', ''),
                                'offset': data.get('offset', ''),
                                'length': data.get('length', '')
                            })
                            
                            if len(all_results) >= limit:
                                break
                            
                except json.JSONDecodeError:
                    continue
            
            if len(all_results) >= limit:
                break
                    
        except requests.exceptions.RequestException as e:
            print(f"  Ошибка при запросе: {e}")
            continue
        except Exception as e:
            print(f"  Неожиданная ошибка: {e}")
            continue
    
    return all_results[:limit]


def fetch_warc_record(filename, offset, length):
    warc_url = f'https://data.commoncrawl.org/{filename}'
    
    headers = {
        'Range': f'bytes={offset}-{int(offset) + int(length) - 1}'
    }
    
    try:
        response = requests.get(warc_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке WARC записи: {e}")
        return None


def extract_text_from_warc(warc_content):
    if not warc_content:
        return None, None
    
    try:
        from warcio.archiveiterator import ArchiveIterator
        from io import BytesIO
        
        stream = BytesIO(warc_content)
        
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                content = record.content_stream().read()
                
                try:
                    html = content.decode('utf-8', errors='ignore')
                except:
                    html = content.decode('latin-1', errors='ignore')
                
                soup = BeautifulSoup(html, 'html.parser')
                
                title = soup.find('title')
                title_text = title.get_text().strip() if title else 'Нет заголовка'
                
                for script in soup(['script', 'style']):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                text_fragment = text[:300] + '...' if len(text) > 300 else text
                
                return title_text, text_fragment
                
    except Exception as e:
        print(f"Ошибка при извлечении текста: {e}")
        return None, None
    
    return None, None


def format_timestamp(timestamp):
    try:
        dt = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp


def display_results(results, show_text=False):
    if not results:
        print("\nРезультаты не найдены.")
        return
    
    print(f"\nНайдено результатов: {len(results)}\n")
    
    table_data = []
    
    for result in tqdm(results, desc="Обработка результатов"):
        url = result['url']
        timestamp = format_timestamp(result['timestamp'])
        
        title = 'N/A'
        text_fragment = 'N/A'
        
        if show_text:
            warc_content = fetch_warc_record(
                result['filename'],
                result['offset'],
                result['length']
            )
            title, text_fragment = extract_text_from_warc(warc_content)
            
            if title is None:
                title = 'Не удалось извлечь'
            if text_fragment is None:
                text_fragment = 'Не удалось извлечь'
        
        row = {
            'URL': url,
            'Дата архивации': timestamp,
            'Заголовок': title
        }
        
        if show_text:
            row['Фрагмент текста'] = text_fragment
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.width', None)
    
    print("\n" + "="*100)
    print(df.to_string(index=True))
    print("="*100 + "\n")


def main():
    args = parse_arguments()
    
    print("\n" + "="*100)
    print("Common Crawl Search Tool")
    print("="*100)
    
    results = search_cdx_index(
        keywords=args.keywords,
        domain=args.domain,
        limit=args.limit
    )
    
    if not results:
        print("\nНичего не найдено. Попробуйте другие ключевые слова или домен.")
        return
    
    display_results(results, show_text=args.show_text)


if __name__ == '__main__':
    main()
