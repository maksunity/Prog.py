import scrapy


class NewsItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    content = scrapy.Field()
    source = scrapy.Field()
    keywords = scrapy.Field()


class ProductItem(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    old_price = scrapy.Field()
    rating = scrapy.Field()
    reviews_count = scrapy.Field()
    url = scrapy.Field()
    image_url = scrapy.Field()
    store = scrapy.Field()
    category = scrapy.Field()
    brand = scrapy.Field()
