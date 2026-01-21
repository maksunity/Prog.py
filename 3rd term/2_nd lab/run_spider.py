import argparse
import sys
import os
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def main():
    parser = argparse.ArgumentParser(description='Web Scraping Tool')
    parser.add_argument('--spider', required=True, choices=['news', 'ecommerce'],
                       help='Spider to run: news or ecommerce')
    parser.add_argument('--keywords', default='', help='Keywords for news search')
    parser.add_argument('--days', type=int, default=7, help='Days to look back for news')
    parser.add_argument('--category', default='ноутбуки', help='Category for ecommerce search')
    parser.add_argument('--stores', default='wildberries', help='Stores for ecommerce (comma-separated)')
    parser.add_argument('--output', required=True, help='Output file (json or csv)')
    
    args = parser.parse_args()
    
    settings = get_project_settings()
    
    process = CrawlerProcess(settings)
    
    if args.spider == 'news':
        print(f'Starting news spider with keywords: {args.keywords}, days: {args.days}')
        print(f'Output file: {args.output}')
        
        process.crawl('news',
                     keywords=args.keywords,
                     days=args.days,
                     output=args.output)
    
    elif args.spider == 'ecommerce':
        print(f'Starting ecommerce spider for category: {args.category}')
        print(f'Stores: {args.stores}')
        print(f'Output file: {args.output}')
        
        process.crawl('ecommerce',
                     category=args.category,
                     stores=args.stores,
                     output=args.output)
    
    process.start()
    
    if os.path.exists(args.output):
        print(f'\nData saved to: {args.output}')
        
        file_size = os.path.getsize(args.output)
        print(f'File size: {file_size} bytes')
    else:
        print('\nWarning: Output file was not created')


if __name__ == '__main__':
    main()
