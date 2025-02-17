import json
import logging
import time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yt_dlp

class DataExtractor:
    content_urls = {
        'black_butler_s1': [
            'https://www.crunchyroll.com/black-butler/episode-1-his-butler-able-786864',
            # ... keeping all existing episodes
        ],
        'black_butler_s2': [
            'https://www.crunchyroll.com/black-butler/episode-1-his-butler-incoming-786907',
            # ... keeping all existing episodes
        ],
        'book_of_circus': [
            'https://www.crunchyroll.com/black-butler-book-of-circus/episode-1-his-butler-recruiting-656619',
            # ... keeping all existing episodes
        ],
        'special_content': {
            'book_of_murder': [
                'https://www.crunchyroll.com/black-butler-book-of-murder/black-butler-book-of-murder-part-1-662943',
                'https://www.crunchyroll.com/black-butler-book-of-murder/black-butler-book-of-murder-part-2-662945'
            ],
            'book_of_atlantic': 'https://www.crunchyroll.com/black-butler-book-of-the-atlantic/black-butler-book-of-the-atlantic-movie-728343',
            'ovas': [
                'https://www.crunchyroll.com/black-butler/episode-1-his-butler-on-stage-special-786919',
                # ... keeping all existing OVAs
            ]
        }
    }

    def __init__(self):
        self.base_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data")
        self.setup_directories()
        self.setup_downloader()
        self.session = self.create_retry_session()

    def setup_directories(self):
        self.datasets = {
            'main': self.base_dir / 'main',
            'context': self.base_dir / 'context',
            'processed': self.base_dir / 'processed',
            'analysis': self.base_dir / 'analysis',
            'video': self.base_dir / 'video',
            'audio': self.base_dir / 'audio',
            'transcripts': self.base_dir / 'transcripts'
        }
        for path in self.datasets.values():
            path.mkdir(parents=True, exist_ok=True)

    def setup_downloader(self):
        self.ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'outtmpl': str(self.datasets['video'] / '%(title)s.%(ext)s'),
            'quiet': True
        }

    def create_retry_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.3)
        adapter = HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def extract_episode_data(self, url):
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            logging.error(f"Failed to extract data from {url}: {str(e)}")
            return None

    def process_content(self):
        print("Starting content extraction")
        
        # Process manga files from main folder
        manga_files = list(self.datasets['main'].glob('*.jpg'))
        for manga_file in manga_files:
            print(f"Processing manga: {manga_file.name}")
            # Add manga processing logic here
            
        # Process wiki content
        wiki_files = list(self.datasets['context'].glob('*.txt'))
        for wiki_file in wiki_files:
            print(f"Processing wiki content: {wiki_file.name}")
            # Add wiki processing logic here
            
        print("Content extraction completed")

if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.process_content()