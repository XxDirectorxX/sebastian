import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import time
from typing import Dict, List, Any, Optional
import torch.nn as nn
import requests

from bs4 import BeautifulSoup
import torch
import numpy as np
from tqdm import tqdm
from pytube import YouTube, Playlist
from yt_dlp import YoutubeDL
import browser_cookie3
import whisper

class ContentProcessor:
    def __init__(self, verbosity='basic'):
        self.setup_logging(verbosity)
        self.num_workers = multiprocessing.cpu_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cookie_path = self._create_cookies_file()

        # Initialize paths with correct base directory
        self.base_path = Path('../..')  # Navigate up to the training directory
        self.datasets = {
            'main': self.base_path / 'raw_data/main',
            'context': self.base_path / 'raw_data/context',
            'processed': self.base_path / 'processed_data',
            'analysis': self.base_path / 'analysis',
            'video': self.base_path / 'raw_data/video',
            'audio': self.base_path / 'raw_data/audio',
            'transcripts': self.base_path / 'raw_data/transcripts'
        }
        
        # Create directories
        for path in self.datasets.values():
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize models
        self.whisper_model = whisper.load_model("base")
        self.trait_tensor = torch.zeros((64, 64, 64), dtype=torch.complex128, device=self.device)
        self.integration_network = nn.Sequential(
            nn.Linear(262144, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 262144)
        ).to(self.device)
        
        self.personality_matrices = self._initialize_personality_matrices()
        
        # Initialize download configuration
        self.cr_config = {
            'format': 'bestvideo[height>=2160]+bestaudio/best[height>=2160]/bestvideo[height>=1080]+bestaudio/best[height>=1080]',
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'embedsubtitles': True,
            'postprocessors': [{
                'key': 'FFmpegEmbedSubtitle'
            }],
            'extractor_args': {
                'crunchyroll': {
                    'premium_only': True,
                    'api_key': self._get_cr_api_key()
                }
            },
            'format_sort': ['res:2160', 'res:1080'],
            'allow_unplayable_formats': True,
            'ignore_no_formats_error': True,
            'merge_output_format': 'mp4'
        }
    
    def _create_cookies_file(self):
        cookie_content = f'''crunchyroll.com    TRUE    /    TRUE    1999999999    etp_rt    {self._get_cr_api_key()}'''
        cookie_path = Path('cookies.txt')
        with open(cookie_path, 'w') as f:
            f.write(cookie_content)
        return cookie_path

    def setup_logging(self, verbosity):
        log_levels = {'basic': logging.INFO, 'detailed': logging.DEBUG}
        log_path = Path('training/data_collection/logs')
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_levels.get(verbosity, logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'extraction_results.txt'),
                logging.StreamHandler()
            ]
        )

    def _initialize_personality_matrices(self):
        return {
            'trait_matrix': torch.randn(64, 64, dtype=torch.float32, device=self.device),
            'interaction_matrix': torch.randn(64, 64, dtype=torch.float32, device=self.device),
            'context_matrix': torch.randn(64, 64, dtype=torch.float32, device=self.device)
        }

    def download_crunchyroll_episodes(self):
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

        with YoutubeDL(self.cr_config) as ydl:
            ydl.add_default_info_extractors()
        
        for season, urls in content_urls.items():
            if isinstance(urls, dict):
                for content_type, content_urls in urls.items():
                    if isinstance(content_urls, list):
                        for url in content_urls:
                            try:
                                logging.info(f"Downloading {content_type} from {url}")
                                ydl.download([url])
                            except Exception as e:
                                logging.error(f"Failed to download {url}: {str(e)}")

    def download_from_specialized_sources(self):
        """Downloads content from specialized trackers"""
        download_config = {
            'format': 'best',
            'merge_output_format': 'mkv',
            'writesubtitles': True,
            'subtitleslangs': ['eng', 'jpn'],
            'postprocessors': [{
                'key': 'FFmpegEmbedSubtitle',
                'already_have_subtitle': False
            }],
            'concurrent_fragments': 5,
            'outtmpl': str(self.datasets['video'] / '%(title)s/%(resolution)s/%(id)s.%(ext)s')
        }

        with YoutubeDL(download_config) as ydl:
            for source_type, urls in self._initialize_anime_seasons()[0].items():
                for url in urls:
                    try:
                        logging.info(f"Downloading from {url}")
                        ydl.download([url])
                    except Exception as e:
                        logging.error(f"Failed to download from {url}: {str(e)}")
                        continue

    def download_manga_content(self):
        """Downloads manga content from all sources with rate limiting and proxy rotation"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        proxies = self._get_proxy_list()
        rate_limit = 2  # seconds between requests
        
        for source in self._initialize_anime_seasons():
            for url in source['manga_chapters']:
                proxy = next(proxies)
                try:
                    time.sleep(rate_limit)
                    response = requests.get(url, headers=headers, proxies={'http': proxy, 'https': proxy})
                    soup = BeautifulSoup(response.content, 'html.parser')
                    manga_data = self._extract_manga_data(soup, url)
                    
                    for chapter in manga_data:
                        output_path = self.datasets['main'] / f"manga_chapter_{chapter['chapter_number']}.json"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(chapter, f, indent=2)
                            
                        # Download images with rate limiting
                        for idx, img_url in enumerate(chapter['images']):
                            time.sleep(rate_limit)
                            img_path = self.datasets['main'] / 'images' / f"chapter_{chapter['chapter_number']}_{idx}.jpg"
                            img_path.parent.mkdir(exist_ok=True)
                            with open(img_path, 'wb') as f:
                                f.write(requests.get(img_url, headers=headers, proxies={'http': proxy, 'https': proxy}).content)
                                
                except Exception as e:
                    logging.error(f"Failed to download from {url}: {str(e)}")
                    continue

    def _get_proxy_list(self):
        """Generates rotating proxy list"""
        proxies = [
            'http://proxy1.example.com:8080',
            'http://proxy2.example.com:8080',
            'http://proxy3.example.com:8080'
        ]
        while True:
            for proxy in proxies:
                yield proxy

    def _get_cr_api_key(self):
        """Get Crunchyroll API key with fallback options"""
        token = 'cbats_H4sIAAAAAAAA_2TKTW6EIBQA4Lu8tSaKqAnn6Kob84Cngog_iFWa3r2Zmcxq9t8vGA0CcMDkZQitmpdU_1ivtwFJx3aGDHA13UR395ReqhBHl3RNZneqrmI1JnvdPX9IpZboj5eknjvpNpxOQmflfVrp-sTOiJCB2gkP0h0eIIAVjOVFkzP-VTaCVaLk35ABXavZKXya9m3-_gMAAP__2N6M378AAAA.MEUCIATMeSR2YnPuF-JaEjzeqBe0uj7461hpNfDzzyYR8_c3AiEAqaqypU-PsFt7camAjlBpHEGV32zOAJ_Xm4N128nNsqo'
        return token
        
    def _get_manual_api_key(self):
        """Fallback method for manual API key entry"""
        logging.info("Please enter your Crunchyroll API key or session token:")
        return input("Enter your API key or session token: ")

    def scrape_content(self):
        """Scrapes content from various sources"""
        sources = self._initialize_anime_seasons()
        scraped_content = []
        
        for source in sources:
            for content_type, urls in source.items():
                for url in urls:
                    content = self._scrape_url(url, content_type)
                    if content:
                        scraped_content.extend(content)
        
        return scraped_content

    def _initialize_anime_seasons(self):
        sources = {
            'character_data': [
                'https://persona-records.fandom.com/wiki/Sebastian_Michaelis',
                'https://anidb.net/character/5832'
            ],
            'anime_sources': [
                'nyaa.si/black-butler-complete',
                'animetosho.org/black-butler',
                'subsplease.org/black-butler',
                'animebytes.tv/black-butler',
                'anidex.info/black-butler',
                'bakabt.me/black-butler',
                'acgnx.se/black-butler'
            ],
            'dub_sources': [
                'dubs.to/anime/black-butler',
                'animedao.to/black-butler',
                'animepahe.com/black-butler',
                'gogoanime.tv/black-butler'
            ],
            'manga_chapters': [
                        'mangadex.org/title/black-butler',
                        'mangasee123.com/manga/Black-Butler',
                        'mangakakalot.com/manga/black_butler',
                        'mangahere.cc/manga/kuroshitsuji',
                        'mangareader.to/black-butler',
                        'manganato.com/manga/black_butler',
                        'fanfox.net/manga/kuroshitsuji'            
            ],
        }
        return [sources]

    def _scrape_url(self, url: str, content_type: str) -> List[Dict]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Referer': 'https://www.crunchyroll.com/'
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        try:
            response = session.get(url)
            response.raise_for_status()
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                if content_type == 'manga_chapters':
                    return self._extract_manga_data(soup, url)
                elif content_type == 'character_data':
                    return self._extract_character_data(soup)
            return []
                
        except Exception as e:
            logging.error(f"Failed to scrape {url}: {e}")
            return []

    def _extract_manga_data(self, soup: BeautifulSoup, source_type: str) -> List[Dict]:
        manga_data = []
        
        if 'mangadex' in source_type:
            chapters = soup.find_all('div', class_='chapter-content')
        elif 'mangasee' in source_type:
            chapters = soup.find_all('div', class_='reader-content')
        elif 'mangakakalot' in source_type:
            chapters = soup.find_all('div', class_='container-chapter-reader')
        
        for chapter in chapters:
            content = chapter.find_all(['p', 'div', 'img'])
            manga_data.append({
                'title': chapter.find('h1', class_='chapter-title').text.strip(),
                'content': '\n'.join([elem.text.strip() for elem in content if elem.text.strip()]),
                'images': [img['src'] for img in chapter.find_all('img')],
                'chapter_number': chapter.get('data-chapter'),
                'type': 'manga'
            })
        
        return manga_data

    def _extract_character_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract character information from BeautifulSoup object"""
        character_data = []
        sections = soup.find_all('div', class_='character-section')
        
        for section in sections:
            character_data.append({
                'name': section.find('h2').text.strip(),
                'description': section.find('div', class_='description').text.strip(),
                'traits': [trait.text for trait in section.find_all('span', class_='trait')],
                'type': 'character'
            })
        return character_data

    def _process_file(self, file_path: Path) -> bool:
        """Process individual content file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            processed_content = self.process_content(content)
            output_path = self.datasets['processed'] / f"processed_{file_path.stem}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_content, f, indent=2)
                
            return True
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return False

    def process_content(self, source_files):
        """Process content through quantum framework"""
        results = {
            'success_count': 0,
            'total_count': len(source_files),
            'success_rate': 0.0,
            'quantum_state_coherence': 1.0
        }
        
        for file_path in source_files:
            try:
                processed = self._process_file(file_path)
                if processed:
                    results['success_count'] += 1
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                
        if results['total_count'] > 0:
            results['success_rate'] = results['success_count'] / results['total_count']
            
        return results

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_files = list(self.base_path.glob('raw_data/temp_*.json'))
            for temp_file in temp_files:
                temp_file.unlink()
            logging.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logging.error(f"Failed to cleanup temp files: {e}")

if __name__ == "__main__":
    processor = ContentProcessor(verbosity='detailed')
    processor.download_crunchyroll_episodes()
    processor.download_from_specialized_sources()
    processor.download_manga_content()
    
     # Process content
    scraped_content = processor.scrape_content()
    
    # Process and analyze
    source_files = []
    for content in scraped_content:
        temp_file = processor.base_path / 'raw_data' / f"temp_{len(source_files)}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        source_files.append(temp_file)
    
    results = processor.process_content(source_files)
    logging.info(f"Processing Results: {json.dumps(results, indent=2)}")
    
    # Clean up temp files
    processor.cleanup_temp_files()

