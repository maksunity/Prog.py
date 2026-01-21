import scrapy
from datetime import datetime, timedelta
from scraper_project.items import NewsItem
import re
from playwright.sync_api import sync_playwright
import time


class NewsSpider(scrapy.Spider):
    name = 'news'
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4,
        'DOWNLOAD_HANDLERS': {},
        'TWISTED_REACTOR': 'twisted.internet.selectreactor.SelectReactor',
    }
    
    def __init__(self, keywords='', days=7, output='news.json', *args, **kwargs):
        super(NewsSpider, self).__init__(*args, **kwargs)
        self.keywords = keywords.split(',') if keywords else []
        self.days = int(days)
        self.output_file = output
        self.items_collected = 0
        self.max_items = 100
        
        self.start_urls = self.get_news_sources()
    
    def get_news_sources(self):
        sources = []
        
        if any('пнипу' in kw.lower() or 'pstu' in kw.lower() or 'perm' in kw.lower() for kw in self.keywords):
            sources.extend([
                'https://pstu.ru/media/news/',
            ])
        
        sources.extend([
            'https://lenta.ru/',
            'https://www.rbc.ru/',
            'https://tass.ru/',
            'https://ria.ru/',
            'https://ria.ru/politics/',
            'https://ria.ru/economy/',
            'https://lenta.ru/rubrics/russia/',
            'https://www.rbc.ru/politics/',
            'https://tass.ru/politika',
        ])
        
        return sources
    
    def parse(self, response):
        try:
            if 'pstu.ru' in response.url:
                # Используем Playwright для динамической подгрузки PSTU
                yield from self.parse_pstu_news_playwright()
            elif 'lenta.ru' in response.url:
                yield from self.parse_lenta_news(response)
            elif 'rbc.ru' in response.url:
                yield from self.parse_rbc_news(response)
            elif 'tass.ru' in response.url or 'ria.ru' in response.url:
                yield from self.parse_generic_news(response)
        except Exception as e:
            self.logger.error(f'Error parsing {response.url}: {str(e)}')
    
    def parse_pstu_news_playwright(self):
        """
        Парсит новости PSTU используя Playwright для динамической подгрузки
        """
        self.logger.info("Запуск парсинга PSTU с Playwright...")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                self.logger.info("Загружаю страницу PSTU...")
                page.goto('https://pstu.ru/media/news/', wait_until='networkidle')
                time.sleep(2)
                
                # Нажимаем кнопку "Показать еще" несколько раз
                clicks = 0
                max_clicks = 15  # Максимум кликов
                
                while clicks < max_clicks and self.items_collected < self.max_items:
                    try:
                        show_more_button = page.locator('text=Показать еще').first
                        
                        if show_more_button.is_visible():
                            self.logger.info(f"Клик #{clicks + 1} на 'Показать еще'...")
                            show_more_button.click()
                            time.sleep(2)
                            clicks += 1
                        else:
                            break
                    except:
                        break
                
                self.logger.info(f"Выполнено {clicks} кликов. Собираю новости...")
                
                # Получаем все ссылки на странице
                all_links = page.locator('a').all()
                
                for link in all_links:
                    if self.items_collected >= self.max_items:
                        break
                    
                    try:
                        href = link.get_attribute('href')
                        
                        if not href or '/media/news/' not in href:
                            continue
                        
                        if href.endswith('/media/news/') or '?tags' in href:
                            continue
                        
                        text = link.inner_text().strip()
                        text = re.sub(r'\s+', ' ', text).strip()
                        text = re.sub(r'\d{1,2}\s+\w+\s+\d{4}', '', text).strip()
                        
                        if len(text) < 15 or len(text) > 300:
                            continue
                        
                        if self.keywords and not self.match_keywords(text):
                            continue
                        
                        # Получаем родительский элемент для поиска даты
                        parent_text = link.locator('xpath=..').inner_text()
                        
                        date_text = datetime.now().strftime('%Y-%m-%d')
                        date_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', parent_text)
                        
                        if date_match:
                            try:
                                months_ru = {
                                    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
                                    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
                                    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
                                }
                                
                                day = int(date_match.group(1))
                                month_name = date_match.group(2).lower()
                                year = int(date_match.group(3))
                                
                                if month_name in months_ru:
                                    month = months_ru[month_name]
                                    date_text = f'{year}-{month:02d}-{day:02d}'
                            except:
                                pass
                        
                        if href.startswith('/'):
                            url = f'https://pstu.ru{href}'
                        else:
                            url = href
                        
                        news_item = NewsItem()
                        news_item['title'] = text
                        news_item['url'] = url
                        news_item['date'] = date_text
                        news_item['source'] = 'ПНИПУ'
                        news_item['keywords'] = ', '.join(self.keywords) if self.keywords else 'general'
                        news_item['content'] = text
                        
                        self.items_collected += 1
                        self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {text[:50]}...')
                        
                        yield news_item
                        
                    except Exception as e:
                        continue
                
                browser.close()
                
        except Exception as e:
            self.logger.error(f'Error in Playwright parsing: {str(e)}')
    
    def parse_pstu_news(self, response):
        all_links = response.css('a')
        
        for link in all_links:
            if self.items_collected >= self.max_items:
                return
            
            try:
                href = link.css('::attr(href)').get()
                
                # Фильтруем только ссылки на новости
                if not href or '/media/news/' not in href:
                    continue
                
                # Пропускаем главную страницу новостей и фильтры по тегам
                if href.endswith('/media/news/') or '?tags' in href:
                    continue
                
                # Получаем полный текст заголовка
                title = ' '.join(link.css('::text').getall()).strip()
                
                # Убираем лишние пробелы и переносы строк
                title = re.sub(r'\s+', ' ', title).strip()
                
                # Убираем даты из заголовка (формат: "22 Января 2026")
                title = re.sub(r'\d{1,2}\s+\w+\s+\d{4}', '', title).strip()
                
                if len(title) < 15 or len(title) > 300:
                    continue
                
                # Проверяем ключевые слова
                if self.keywords and not self.match_keywords(title):
                    continue
                
                # Получаем родительский элемент для поиска даты
                parent = link.xpath('..')
                all_text_in_parent = ' '.join(parent.css('::text').getall())
                
                # Ищем дату в формате "22 Января 2026"
                date_text = datetime.now().strftime('%Y-%m-%d')
                date_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', all_text_in_parent)
                if date_match:
                    try:
                        # Словарь для преобразования месяцев
                        months_ru = {
                            'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
                            'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
                            'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
                        }
                        
                        day = int(date_match.group(1))
                        month_name = date_match.group(2).lower()
                        year = int(date_match.group(3))
                        
                        if month_name in months_ru:
                            month = months_ru[month_name]
                            date_text = f'{year}-{month:02d}-{day:02d}'
                    except:
                        pass
                
                url = response.urljoin(href)
                
                news_item = NewsItem()
                news_item['title'] = title
                news_item['url'] = url
                news_item['date'] = date_text
                news_item['source'] = 'ПНИПУ'
                news_item['keywords'] = ', '.join(self.keywords) if self.keywords else 'general'
                news_item['content'] = title
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield news_item
                
            except Exception as e:
                self.logger.debug(f'Error parsing PSTU item: {str(e)}')
                continue
    
    def parse_lenta_news(self, response):
        links = response.css('a')
        
        for link in links[:50]:
            if self.items_collected >= self.max_items:
                return
            
            try:
                title = link.css('::text').get()
                url = link.css('::attr(href)').get()
                
                if not title or not url:
                    continue
                
                title = title.strip()
                if len(title) < 15 or len(title) > 200:
                    continue
                
                url = response.urljoin(url)
                
                if self.keywords and not self.match_keywords(title):
                    continue
                
                news_item = NewsItem()
                news_item['title'] = title
                news_item['url'] = url
                news_item['date'] = datetime.now().strftime('%Y-%m-%d')
                news_item['source'] = 'Lenta.ru'
                news_item['keywords'] = ', '.join(self.keywords) if self.keywords else 'general'
                news_item['content'] = title
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield news_item
                
            except Exception as e:
                self.logger.debug(f'Error parsing Lenta item: {str(e)}')
                continue
    
    def parse_rbc_news(self, response):
        links = response.css('a')
        
        for link in links[:50]:
            if self.items_collected >= self.max_items:
                return
            
            try:
                title = link.css('::text').get()
                url = link.css('::attr(href)').get()
                
                if not title or not url:
                    continue
                
                title = title.strip()
                if len(title) < 15 or len(title) > 200:
                    continue
                
                url = response.urljoin(url)
                
                if self.keywords and not self.match_keywords(title):
                    continue
                
                news_item = NewsItem()
                news_item['title'] = title
                news_item['url'] = url
                news_item['date'] = datetime.now().strftime('%Y-%m-%d')
                news_item['source'] = 'RBC.ru'
                news_item['keywords'] = ', '.join(self.keywords) if self.keywords else 'general'
                news_item['content'] = title
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield news_item
                
            except Exception as e:
                self.logger.debug(f'Error parsing RBC item: {str(e)}')
                continue
    
    def parse_generic_news(self, response):
        links = response.css('a')
        
        for link in links[:50]:
            if self.items_collected >= self.max_items:
                return
            
            try:
                title = link.css('::text').get()
                url = link.css('::attr(href)').get()
                
                if not title or not url:
                    continue
                
                title = title.strip()
                if len(title) < 15 or len(title) > 200:
                    continue
                
                url = response.urljoin(url)
                
                if self.keywords and not self.match_keywords(title):
                    continue
                
                source_name = 'News'
                if 'tass.ru' in response.url:
                    source_name = 'TASS'
                elif 'ria.ru' in response.url:
                    source_name = 'RIA'
                
                news_item = NewsItem()
                news_item['title'] = title
                news_item['url'] = url
                news_item['date'] = datetime.now().strftime('%Y-%m-%d')
                news_item['source'] = source_name
                news_item['keywords'] = ', '.join(self.keywords) if self.keywords else 'general'
                news_item['content'] = title
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield news_item
                
            except Exception as e:
                self.logger.debug(f'Error parsing generic item: {str(e)}')
                continue
    
    def match_keywords(self, text):
        if not self.keywords:
            return True
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)
