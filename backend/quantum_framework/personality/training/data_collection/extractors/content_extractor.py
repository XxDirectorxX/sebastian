from pathlib import Path
import logging
import asyncio
from yt_dlp import YoutubeDL
import json
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class DataExtractor:
    def __init__(self):
        self.base_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data")
        self.datasets = {
            'main': self.base_dir / 'main',
            'context': self.base_dir / 'context',
            'processed': self.base_dir / 'processed',
            'analysis': self.base_dir / 'analysis',
            'video': self.base_dir / 'video',
            'audio': self.base_dir / 'audio',
            'transcripts': self.base_dir / 'transcripts'
        }
        self.setup_logging()
        self.setup_downloader()
        self.session = self.create_session()

    def setup_logging(self):
        log_file = Path('extraction_log.txt')
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info("=== Data Extraction Started ===")
        logging.info(f"Base Directory: {self.base_dir}")
        logging.info(f"Log File: {log_file.absolute()}")

    def setup_downloader(self):
        self.ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            'username': 'isabel.sustaita06@gmail.com',
            'password': 'Id10t!',
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'embedsubtitles': True,
            'outtmpl': str(self.datasets['video'] / '%(series)s/%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegEmbedSubtitle'
            }]
        }

    def create_session(self):
        session = requests.Session()
        retry = Retry(total=3)
        adapter = HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def download_episodes(self):
        content_urls = {
            'black_butler_s1': [
                'https://www.crunchyroll.com/black-butler/episode-1-his-butler-able-786864',
                'https://www.crunchyroll.com/black-butler/episode-2-his-butler-strongest-786865',
                'https://www.crunchyroll.com/black-butler/episode-3-his-butler-omnipotent-786866',
                'https://www.crunchyroll.com/black-butler/episode-4-his-butler-capricious-786867',
                'https://www.crunchyroll.com/black-butler/episode-5-his-butler-chance-encounter-786868',
                'https://www.crunchyroll.com/black-butler/episode-6-his-butler-at-the-funeral-786869',
                'https://www.crunchyroll.com/black-butler/episode-7-his-butler-merrymaking-786870',
                'https://www.crunchyroll.com/black-butler/episode-8-his-butler-training-786871',
                'https://www.crunchyroll.com/black-butler/episode-9-his-butler-phantom-image-786872',
                'https://www.crunchyroll.com/black-butler/episode-10-his-butler-on-ice-786873',
                'https://www.crunchyroll.com/black-butler/episode-11-his-butler-however-you-please-786874',
                'https://www.crunchyroll.com/black-butler/episode-12-his-butler-forlorn-786875',
                'https://www.crunchyroll.com/black-butler/episode-13-his-butler-freeloader-786876',
                'https://www.crunchyroll.com/black-butler/episode-14-his-butler-supremely-talented-786877',
                'https://www.crunchyroll.com/black-butler/episode-15-his-butler-in-an-isolated-castle-786878',
                'https://www.crunchyroll.com/black-butler/episode-16-his-butler-in-a-festival-786879',
                'https://www.crunchyroll.com/black-butler/episode-17-his-butler-offering-786880',
                'https://www.crunchyroll.com/black-butler/episode-18-his-butler-transmitted-786881',
                'https://www.crunchyroll.com/black-butler/episode-19-his-butler-imprisoned-786882',
                'https://www.crunchyroll.com/black-butler/episode-20-his-butler-escaped-786883'
            ],
            'black_butler_s2': [
                'https://www.crunchyroll.com/black-butler/episode-1-his-butler-incoming-786907',
                'https://www.crunchyroll.com/black-butler/episode-2-his-butler-taking-the-stage-786908',
                'https://www.crunchyroll.com/black-butler/episode-3-his-butler-entertaining-786909',
                'https://www.crunchyroll.com/black-butler/episode-4-his-butler-teaching-786910',
                'https://www.crunchyroll.com/black-butler/episode-5-his-butler-disturbing-786911',
                'https://www.crunchyroll.com/black-butler/episode-6-his-butler-releasing-786912',
                'https://www.crunchyroll.com/black-butler/episode-7-his-butler-lurking-786913',
                'https://www.crunchyroll.com/black-butler/episode-8-his-butler-manipulating-786914',
                'https://www.crunchyroll.com/black-butler/episode-9-his-butler-a-performers-life-786915',
                'https://www.crunchyroll.com/black-butler/episode-10-his-butler-on-ice-786916',
                'https://www.crunchyroll.com/black-butler/episode-11-his-butler-engaging-servants-786917',
                'https://www.crunchyroll.com/black-butler/episode-12-his-butler-finalizing-786918'
            ],
            'book_of_circus': [
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-1-his-butler-recruiting-656619',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-2-his-butler-taking-flight-656621',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-3-his-butler-teaching-656623',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-4-his-butler-showing-off-656625',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-5-his-butler-collecting-656627',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-6-his-butler-infiltrating-656629',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-7-his-butler-lying-in-wait-656631',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-8-his-butler-chasing-656633',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-9-his-butler-competing-656635',
                'https://www.crunchyroll.com/black-butler-book-of-circus/episode-10-his-butler-presenting-656637'
            ],
            'special_content': {
                'book_of_murder': [
                    'https://www.crunchyroll.com/black-butler-book-of-murder/black-butler-book-of-murder-part-1-662943',
                    'https://www.crunchyroll.com/black-butler-book-of-murder/black-butler-book-of-murder-part-2-662945'
                ],
                'book_of_atlantic': 'https://www.crunchyroll.com/black-butler-book-of-the-atlantic/black-butler-book-of-the-atlantic-movie-728343',
                'ovas': [
                    'https://www.crunchyroll.com/black-butler/episode-1-his-butler-on-stage-special-786919',
                    'https://www.crunchyroll.com/black-butler/episode-2-ciel-in-wonderland-part-1-special-786920',
                    'https://www.crunchyroll.com/black-butler/episode-3-ciel-in-wonderland-part-2-special-786921',
                    'https://www.crunchyroll.com/black-butler/episode-4-welcome-to-the-phantomhives-special-786922',
                    'https://www.crunchyroll.com/black-butler/episode-5-making-of-black-butler-2-special-786923',
                    'https://www.crunchyroll.com/black-butler/episode-6-the-threads-of-the-spiders-story-special-786924',
                    'https://www.crunchyroll.com/black-butler/episode-7-the-story-of-will-the-reaper-special-786925'
                ]
            }
        }
        self.manga_sources = [
            'mangadex.org/title/black-butler',
            'mangasee123.com/manga/Black-Butler',
            'mangakakalot.com/manga/black_butler'
        ]

        self.character_sources = [
            'kuroshitsuji.fandom.com/wiki/Sebastian_Michaelis',
            'myanimelist.net/character/4963/Sebastian_Michaelis',
            'www.behindthevoiceactors.com/characters/Black-Butler/Sebastian-Michaelis/'
        ]

    def setup_directories(self):
        self.dirs = {
            'raw_data': self.base_dir / "training/data_collection/raw_data",
            'processed_data': self.base_dir / "training/data_collection/processed_data",
            'logs': self.base_dir / "training/data_collection/logs",
            'video': self.base_dir / "training/data_collection/raw_data/video/Sebastian Michaelis",
            'audio': self.base_dir / "training/data_collection/raw_data/audio/episodes",
            'context': self.base_dir / "training/data_collection/raw_data/context",
            'transcripts': self.base_dir / "training/data_collection/raw_data/transcripts",
            'dialogues': self.base_dir / "training/data_collection/raw_data/text/dialogues",
            'subtitles': self.base_dir / "training/data_collection/raw_data/text/subtitles"
        }

        with YoutubeDL(self.ydl_opts) as ydl:
                for season, episodes in content_urls.items():
                    logging.info(f"Downloading {season}")
                    for url in episodes:
                        try:
                            logging.info(f"Downloading: {url}")
                            ydl.download([url])
                        except Exception as e:
                            logging.error(f"Failed to download {url}: {str(e)}")

    def extract_wiki_data(self):
        url = 'https://kuroshitsuji.fandom.com/wiki/Sebastian_Michaelis'
        logging.info(f"Extracting data from: {url}")
        
        response = self.session.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            data = {
                'character_info': [],
                'personality': [],
                'quotes': [],
                'relationships': [],
                'abilities': []
            }
            
            # Extract data using BeautifulSoup selectors
            # Save to JSON in main dataset directory
            output_path = self.datasets['main'] / 'sebastian_data.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def extract_manga_data(self):
        manga_urls = [
            'https://mangadex.org/title/8bd19e5c-94f7-4368-a918-50f463857446/kuroshitsuji'
        ]
        
        for url in manga_urls:
            logging.info(f"Extracting manga data from: {url}")
            # Extract manga content
            # Save to JSON in main dataset directory

    def process_content(self):
        logging.info("Starting content extraction")
        self.download_episodes()
        self.extract_wiki_data()
        self.extract_manga_data()
        logging.info("Content extraction completed")

if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.process_content()