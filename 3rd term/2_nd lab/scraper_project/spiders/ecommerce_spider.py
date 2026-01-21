import scrapy
from scraper_project.items import ProductItem
import json
import re


class EcommerceSpider(scrapy.Spider):
    name = 'ecommerce'
    
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_HANDLERS': {
            'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
        },
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'PLAYWRIGHT_ABORT_REQUEST': lambda req: req.resource_type in ['image', 'stylesheet', 'font'],
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000,
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 60000,
            'args': ['--disable-blink-features=AutomationControlled'],
        },
    }
    
    def __init__(self, category='ноутбуки', stores='wildberries,ozon', output='products.json', *args, **kwargs):
        super(EcommerceSpider, self).__init__(*args, **kwargs)
        self.category = category
        self.stores = stores.split(',') if stores else ['wildberries']
        self.output_file = output
        self.items_collected = 0
        self.max_items = 100
        
        self.start_urls = []
        self.setup_start_urls()
    
    def setup_start_urls(self):
        if 'wildberries' in self.stores:
            self.start_urls.append(f'https://www.wildberries.ru/catalog/0/search.aspx?search={self.category}')
        
        if 'ozon' in self.stores:
            self.start_urls.append(f'https://www.ozon.ru/search/?text={self.category}&from_global=true')
    
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={
                    'playwright': True,
                    'playwright_include_page': True,
                    'playwright_page_methods': [
                        {'method': 'wait_for_load_state', 'kwargs': {'state': 'domcontentloaded', 'timeout': 60000}},
                    ],
                },
                errback=self.errback_close_page,
            )
    
    async def parse(self, response):
        page = response.meta.get('playwright_page')
        
        try:
            if 'wildberries.ru' in response.url:
                async for item in self.parse_wildberries(response, page):
                    yield item
            elif 'ozon.ru' in response.url:
                async for item in self.parse_ozon(response, page):
                    yield item
        finally:
            if page:
                await page.close()
    
    async def parse_wildberries(self, response, page):
        try:
            if page:
                await page.wait_for_selector('.product-card, article.product-card, div[data-goods-id]', timeout=10000)
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight / 2)')
                await page.wait_for_timeout(2000)
        except Exception as e:
            self.logger.warning(f'Timeout waiting for products: {e}')
        
        products = response.css('.product-card, article.product-card, div[data-goods-id]')[:50]
        
        for product in products:
            if self.items_collected >= self.max_items:
                return
            
            try:
                title_elem = product.css(
                    'div.product-card__name::text, '
                    'a.product-card__link::text, '
                    'span.goods-name::text, '
                    'h2::text, '
                    'h3::text'
                )
                title = title_elem.get()
                
                if not title:
                    continue
                
                title = title.strip()
                
                price_elem = product.css(
                    'span.price__lower-price::text, '
                    'ins.price-block__final-price::text, '
                    'span.price-commission::text, '
                    'span[class*="price"]::text'
                )
                price_text = price_elem.get()
                price = self.extract_price(price_text) if price_text else '0'
                
                old_price_elem = product.css(
                    'del.price-block__old-price::text, '
                    'span.price__del::text, '
                    'del::text'
                )
                old_price_text = old_price_elem.get()
                old_price = self.extract_price(old_price_text) if old_price_text else price
                
                rating = product.css('span.product-card__rating::text, span.address-rate-mini::text').get()
                reviews = product.css('span.product-card__count::text, span.product-card__count-review::text').get()
                
                url = product.css('a::attr(href)').get()
                if url:
                    url = response.urljoin(url)
                
                image_url = product.css('img::attr(src), img::attr(data-src)').get()
                
                product_item = ProductItem()
                product_item['title'] = title
                product_item['price'] = price
                product_item['old_price'] = old_price
                product_item['rating'] = rating.strip() if rating else 'N/A'
                product_item['reviews_count'] = reviews.strip() if reviews else '0'
                product_item['url'] = url if url else response.url
                product_item['image_url'] = image_url if image_url else ''
                product_item['store'] = 'Wildberries'
                product_item['category'] = self.category
                product_item['brand'] = 'Unknown'
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield product_item
            
            except Exception as e:
                self.logger.debug(f'Error parsing Wildberries product: {str(e)}')
                continue
    
    async def parse_ozon(self, response, page):
        try:
            if page:
                await page.wait_for_selector('div.tile-root, div[data-widget="searchResultsV2"]', timeout=5000)
        except:
            self.logger.warning('Timeout waiting for products')
        
        products = response.css('div.tile-root, div.widget-search-result-container div')[:50]
        
        for product in products:
            if self.items_collected >= self.max_items:
                return
            
            try:
                title = product.css('span.tsBody500Medium::text, a.tile-hover-target span::text').get()
                
                if not title:
                    continue
                
                title = title.strip()
                
                price_text = product.css('span.c3017-a1::text, span[class*="price"]::text').get()
                price = self.extract_price(price_text) if price_text else '0'
                
                old_price_text = product.css('span.c3017-a7::text').get()
                old_price = self.extract_price(old_price_text) if old_price_text else price
                
                rating = product.css('div.tsBodyControl400Small::text').get()
                reviews = product.css('span[class*="reviews"]::text').get()
                
                url = product.css('a::attr(href)').get()
                if url:
                    url = response.urljoin(url)
                
                image_url = product.css('img::attr(src)').get()
                
                product_item = ProductItem()
                product_item['title'] = title
                product_item['price'] = price
                product_item['old_price'] = old_price
                product_item['rating'] = rating.strip() if rating else 'N/A'
                product_item['reviews_count'] = reviews.strip() if reviews else '0'
                product_item['url'] = url if url else response.url
                product_item['image_url'] = image_url if image_url else ''
                product_item['store'] = 'Ozon'
                product_item['category'] = self.category
                product_item['brand'] = 'Unknown'
                
                self.items_collected += 1
                self.logger.info(f'Collected {self.items_collected}/{self.max_items}: {title[:50]}...')
                
                yield product_item
            
            except Exception as e:
                self.logger.debug(f'Error parsing Ozon product: {str(e)}')
                continue
    
    def extract_price(self, price_text):
        if not price_text:
            return '0'
        
        price_text = price_text.strip()
        digits = re.findall(r'\d+', price_text)
        
        if digits:
            return ''.join(digits)
        
        return '0'
    
    async def errback_close_page(self, failure):
        page = failure.request.meta.get('playwright_page')
        if page:
            await page.close()
        
        self.logger.error(f'Request failed: {failure.request.url}')
        self.logger.error(f'Error: {failure.value}')
