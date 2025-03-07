# %%
import os
import sys
import json
import time
import socket
import logging
import argparse
import requests
import langchain
import threading
import webbrowser
import urllib.parse
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel
from urllib.parse import urlparse
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
from typing import List, Dict, Optional, Tuple, Set

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException
from http.server import BaseHTTPRequestHandler, HTTPServer
from tenacity import retry, stop_after_attempt, wait_exponential

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
parser = argparse.ArgumentParser(description='Scrape music data from a YouTube channel')
parser.add_argument('--url', type=str, help='URL to scrape chat data from')
parser.add_argument('--full', type=bool, default=False, help='Scrape all videos data if True')
parser.add_argument('--output', type=str, default='report.xlsx', help='Output file path')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')

# %%
def get_file_from_gist(gist_id: str, gist_token: str, file_name: str) -> Optional[Dict]:
    headers = {'Authorization': f'token {gist_token}'}
    url = f'https://api.github.com/gists/{gist_id}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        files = data.get('files', {})

        if file_name in files:
            return json.loads(files[file_name]['content'])
        else:
            return {}

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao fazer a requisição: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao decodificar o JSON: {e}")
    except KeyError as e:
        logging.error(f"Erro: Chave ausente na resposta: {e}")

    return None

# %%
class App():
    def __init__(self, channel_url: str, full: bool, output: str, temperature: float):            
        self.channel_url = channel_url
        self.channel_name = channel_url.rsplit("/")[-2].replace("@", "")
        self.full = full
        self.output = output
        self.database = self._start_database()
        self.titles = self._get_content()
        self.model_name = "llama-3.3-70b-versatile"
        self.temperature = temperature
        self.chat = self._instance_model()
        self.user_id = os.environ['SPOTIFY_USER_ID']
        self.redirect_uri = os.environ.get('SPOTIFY_REDIRECT_URI')
        self.client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        self.spotify_api = 'https://api.spotify.com/v1'
        self.spotify_auth_code = self._get_spotify_auth_code()
        self.spotify_access_token = self._get_spotify_access_token()
        self.headers = {'Authorization': f'Bearer {self.spotify_access_token}'}
        self.playlist_url = f'{self.spotify_api}/users/{self.user_id}/playlists'
        self.playlist_id = self._get_playlist_id()
        self.gist_token = os.environ.get('GIST_ACCESS_TOKEN')
        self.now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    
    def _instance_model(self) -> callable:
        class MusicDetails(BaseModel):
            artist: str
            track: str
            title: str
            
        llm = ChatGroq(model_name=self.model_name, temperature=self.temperature)
        parser = JsonOutputParser(pydantic_object=MusicDetails)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """"You are a JSON extraction assistant. Always respond with a valid JSON using the structure below.\n
            If there are no explicit mentions of a song (artist name and/or track title), return unknown for the fields.

                {{
                    "artist": "artist name here",
                    "track": "track name here",
                    "title": "full title here, artist + track"
                }}"""),
            ("user", "{input}")
        ])
        
        return prompt | llm | parser
       
       
    def _start_database(self) -> callable:
        uri = os.getenv('DATABASE_URI')

        client = MongoClient(uri, server_api=ServerApi('1'))

        try:
            client.admin.command('ping')
        except Exception as e:
            logging.error("Falha ao se conectar com o banco de dados")
            sys.exit()

        db = client['music-scrapping']
        
        return db['report']


    def _identify(self, title: str) -> bool:
        if self.database.find_one({'original_title': title}):
            return True
        return False
        

    def _parse(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        titles_elements = soup.find_all(id="video-title")
    
        titles = [title.get_text(strip=True) for title in titles_elements]
        
        return [title for title in titles if not self._identify(title)]


    def _get_content(self) -> List[str]:
        logging.info('Iniciando extração de títulos...')

        if not self.channel_url.startswith('http'):
            raise ValueError(f"Endereço inválido: {self.channel_url}")

        options = Options()
        options.add_argument("--headless")

        driver = None
        try:
            driver = webdriver.Firefox(options=options)
            driver.get(self.channel_url)
            logging.info('Página carregada.')

            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            if self.full:
                last_height = driver.execute_script("return document.documentElement.scrollHeight")

                while True:
                    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                    
                    try:
                        WebDriverWait(driver, 2).until(
                            lambda d: d.execute_script("return document.documentElement.scrollHeight") > last_height
                        )
                    except Exception:
                        break

                    last_height = driver.execute_script("return document.documentElement.scrollHeight")

            html = driver.page_source
            logging.info('Extração concluída.')

            return self._parse(html)

        except Exception as e:
            logging.error(f"Erro ao obter conteúdo: {e}")
            return []

        finally:
            if driver:
                driver.quit()
    
    
    def _get_wait_time(self, error: str) -> int:
        try:
            seconds = str(error).split('Please try again in ')[-1].split('s', 1)[0]
            seconds = float(seconds.replace('.', '').replace('m', '.'))
            return int(seconds * 60)
        except Exception:
            logging.warning("Falha ao extrair tempo de espera da mensagem de erro.")
            return 120
    
    
    def _ask(self, description: str) -> Dict[str, str]:
        default_response = {"artist": "Unknown", "track": "Unknown", "title": "Unknown"}

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
        def _attempt() -> Dict[str, str]:
            return self.chat.invoke(input=description)

        try:
            return _attempt()
        except OutputParserException:
            logging.warning(f"Falha ao converter resposta: {description}")
        except Exception as error:
            wait_time = self._get_wait_time(str(error))
            logging.error(f"Nova tentativa em {wait_time} seconds.")
            time.sleep(wait_time)
            try:
                return self.chat.invoke(input=description)
            except Exception as final_error:
                logging.error(f"Extração dados falhou {description}. Erro: {final_error}")

        return default_response
            
            
    def _get_new_tracks(self) -> List[Dict[str, str]]:
        new_tracks = []
        
        for title in tqdm(self.titles, desc="Extraindo", ncols=80):
            music_obj = self._ask(title)

            for key in music_obj.keys():
                music_obj[key] = music_obj[key].title()

            music_obj['original_title'] = title
            music_obj['channel'] = self.channel_name
            music_obj['_date'] = self.now
            
            new_tracks.append(music_obj)
        
        return new_tracks
    

    def _get_spotify_auth_code(self) -> str:
        url = 'https://accounts.spotify.com/authorize'
        port = urlparse(self.redirect_uri).port
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': "user-read-private user-read-email playlist-modify-public playlist-modify-private"
        }

        authorization_url = f"{url}?{urllib.parse.urlencode(params)}"
        
        webbrowser.open(authorization_url)

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                if 'code' in params:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Autorizacao concluida! Pode fechar esta janela.")
                    self.server.auth_code = params['code'][0]
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Codigo de autorizacao nao encontrado.")

        server = HTTPServer(('localhost', port), RequestHandler)
        
        def run_server() -> None:
            logging.info("Aguardando autorizacao...")
            server.handle_request()

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        server_thread.join(timeout=10)

        auth_code = getattr(server, 'auth_code', None)

        if not auth_code:
            logging.warning("Tempo limite atingido! Insira manualmente o código de autorização:")
            auth_code = input("Insira o código/link: ").strip()
            
            if auth_code.startswith('http'):
                auth_code = auth_code.split('=', 1)[-1]
        
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                pass
        except (ConnectionRefusedError, OSError):
            pass

        return auth_code
    
    
    def _get_spotify_access_token(self) -> str:
        url = 'https://accounts.spotify.com/api/token'

        data = {
            'grant_type': 'authorization_code',
            'code': self.spotify_auth_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }

        return requests.post(url, data=data).json().get('access_token')
        

    def _get_playlist_id(self) -> str:
        user_playlists = requests.get(self.playlist_url, headers=self.headers).json()

        for playlist in user_playlists['items']:
            if 'Youtube Scrapping' in playlist['name']:
                return playlist['id']
            else:
                return requests.post(self.playlist_url, headers=self.headers, json={
                    'name': 'Youtube Scrapping',
                    'description': 'Musicas que foram retiradas do Youtube',
                    'public': True
                }).json()['id']

    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _update_database(self, report_data: List[Dict[str, str]]) -> None:
        if report_data:
            try:
                self.database.insert_many(report_data)
            except Exception as e:
                logging.error(f"Falha ao atualizar o banco de dados: {e}")
                raise
        
        
    def _get_track_metadata(self, artist: str, music: str) -> Optional[Tuple[str, str, str]]:
        try:
            response = requests.get(
                f'{self.spotify_api}/search',
                headers=self.headers,
                params={"q": f"artist:{artist} track:{music}", "type": "track"}
            )
            response.raise_for_status()
            track_metadata = response.json()['tracks']['items']

            if track_metadata:
                track_info = track_metadata[0]
                return track_info['uri'], track_info['name'], track_info['artists'][0]['name']
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao localizar informações da faixa: {e}")
            return None
    
    
    def _get_existing_tracks(self, url: str) -> Set[Tuple[str, str]]:
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            playlist = response.json()['items']
            return {(item['track']['name'], item['track']['artists'][0]['name']) for item in playlist}
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao localizar faixas na playlist: {e}")
            return set()
    

    def _add_track_to_playlist(self, url: str, track_uri: str) -> bool:
        try:
            response = requests.post(url, headers=self.headers, json={'uris': [track_uri]})
            response.raise_for_status()
            return response.status_code == 201
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao adicionar faixa a playlist: {e}")
            return False
        
        
    def _save_not_added_tracks(self, not_added_tracks: list) -> None:
        if not_added_tracks:
            df = pd.DataFrame(not_added_tracks)
            df.to_excel(self.output, index=False)
            logging.info(f'Não foi possível identificar {len(not_added_tracks)} títulos. Salvo em {self.output}')
        

    def run(self):
        new_tracks = self._get_new_tracks()
        url = f'{self.spotify_api}/playlists/{self.playlist_id}/tracks'
        not_added_tracks = []
        report_data = []
        
        for track in tqdm(new_tracks, desc="Adicionando a playlist:", ncols=80):
            artist = track['artist']
            music = track['track']
            
            try:
                report_data.append(track)

                track_metadata = self._get_track_metadata(artist, music)
                
                if not track_metadata:
                    not_added_tracks.append(track)
                    continue

                track_uri, new_track, new_artist = track_metadata
                existing_tracks = self._get_existing_tracks(url)

                if (new_track, new_artist) not in existing_tracks:
                    if not self._add_track_to_playlist(url, track_uri):
                        logging.error(f"Falha ao adicionar faixa: {new_track} - {new_artist}.")
                        not_added_tracks.append(track)
                
            except Exception as e:
                logging.error(f"Erro ao processar faixa {music} - {artist}: {e}")
                not_added_tracks.append(track)

        self._save_not_added_tracks(not_added_tracks)
        self._update_database(report_data)
        
        logging.info('Execução finalizada!')

# %%
load_dotenv()

GIST_ID = os.getenv('GIST_ID')
GIST_TOKEN = os.getenv('GIST_TOKEN')
GIST_FILE = "youtube-music-scrapping.json"

credentials = get_file_from_gist(GIST_ID, GIST_TOKEN, GIST_FILE)

os.environ['SPOTIFY_USER_ID'] = credentials['user_id']
os.environ['SPOTIFY_CLIENT_ID'] = credentials['client_id']
os.environ['SPOTIFY_CLIENT_SECRET'] = credentials['client_secret']
os.environ['SPOTIFY_REDIRECT_URI'] = credentials['redirect_uri']
os.environ['GROQ_API_KEY'] = credentials['groq_api_key']

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    
    app = App(
        channel_url=args.url,
        full=args.full,
        output=args.output,
        temperature=args.temperature
    )
    
    app.run()


