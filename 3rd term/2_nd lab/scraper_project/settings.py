BOT_NAME = 'scraper_project'

SPIDER_MODULES = ['scraper_project.spiders']
NEWSPIDER_MODULE = 'scraper_project.spiders'

ROBOTSTXT_OBEY = False

CONCURRENT_REQUESTS = 16
DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 8

COOKIES_ENABLED = True

TELNETCONSOLE_ENABLED = False

DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

DOWNLOAD_HANDLERS = {
    'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
    'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
}

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
}

ITEM_PIPELINES = {
    'scraper_project.pipelines.DataCleaningPipeline': 300,
    'scraper_project.pipelines.JsonWriterPipeline': 800,
    'scraper_project.pipelines.CsvWriterPipeline': 801,
}

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

PLAYWRIGHT_BROWSER_TYPE = 'chromium'
PLAYWRIGHT_LAUNCH_OPTIONS = {
    'headless': True,
    'timeout': 30000,
}

PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000

LOG_LEVEL = 'INFO'
