import json
import csv
import os
from datetime import datetime


class JsonWriterPipeline:
    
    def open_spider(self, spider):
        if hasattr(spider, 'output_file') and spider.output_file:
            if spider.output_file.endswith('.json'):
                self.file = open(spider.output_file, 'w', encoding='utf-8')
                self.items = []
    
    def close_spider(self, spider):
        if hasattr(self, 'file'):
            json.dump(self.items, self.file, ensure_ascii=False, indent=2)
            self.file.close()
    
    def process_item(self, item, spider):
        if hasattr(self, 'file'):
            self.items.append(dict(item))
        return item


class CsvWriterPipeline:
    
    def open_spider(self, spider):
        if hasattr(spider, 'output_file') and spider.output_file:
            if spider.output_file.endswith('.csv'):
                self.file = open(spider.output_file, 'w', encoding='utf-8', newline='')
                self.writer = None
    
    def close_spider(self, spider):
        if hasattr(self, 'file'):
            self.file.close()
    
    def process_item(self, item, spider):
        if hasattr(self, 'file'):
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=item.keys())
                self.writer.writeheader()
            self.writer.writerow(dict(item))
        return item


class DataCleaningPipeline:
    
    def process_item(self, item, spider):
        for field, value in item.items():
            if isinstance(value, str):
                item[field] = value.strip()
        return item
