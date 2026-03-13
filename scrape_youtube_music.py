# %%
import os
import re
import time
import logging
import argparse
import requests

from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, Tuple, Set

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException
from tenacity import retry, stop_after_attempt, wait_exponential

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
parser = argparse.ArgumentParser(description='Scrape music data from a YouTube channel')
parser.add_argument('--url', type=str, help='URL to scrape chat data from')
parser.add_argument('--full', type=bool, default=False, help='Scrape all videos data if True')
parser.add_argument('--model_name', type=str, help='IA Model')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')

# %%
class App():
    def __init__(self, channel_url: str, full: bool, model_name: str, temperature: float):
        """
        Inicializa a aplicação com as configurações necessárias.

        Args:
            channel_url (str): URL do canal.
            full (bool): Flag para carregar modo completo.
            model_name (str): Nome do modelo a ser utilizado.
            temperature (float): Temperatura do modelo (parâmetro de criatividade).
        """
        #Parametros de entrada
        self.channel_url = channel_url
        self.full = full
        self.model_name = model_name
        self.temperature = temperature
        
        # Parametros de configuração
        self.channel_name = channel_url.rsplit("@")[-1]
        self.days = 999999 if full else 30
        
        # Carrega variáveis de ambiente necessárias
        self.client_id = os.environ.get("CLIENT_ID")
        self.user_id = os.environ.get("USER_ID")
        self.client_secret = os.environ.get("CLIENT_SECRET")
        self.refresh_token = os.environ.get("REFRESH_TOKEN")
        self.yt_api_key = os.environ.get("YT_API_KEY")
        
        # Endpoints da API
        self.spotify_api = "https://api.spotify.com/v1"
        self.youtube_api = "https://www.googleapis.com/youtube/v3"
        
        # Instancia modelos e tokens
        self.chat = self._instance_model()
        self.spotify_token = self._get_spotify_access_token()
        self.headers = {"Authorization": f"Bearer {self.spotify_token}"}
        self.playlist_id = self._get_playlist_id()
        self.playlist_url = f'{self.spotify_api}/playlists/{self.playlist_id}/tracks'
        
    def _instance_model(self) -> callable:
        """
        Cria e retorna um pipeline de extração automática de dados musicais estruturados (artista, faixa, título ou álbum) 
        utilizando um modelo de linguagem natural.

        O prompt orienta o modelo a:
            - Extrair informações sobre músicas ou álbuns a partir de descrições em texto livre.
            - Sempre gerar uma resposta no formato JSON.
            - Preencher campos como "unknown" caso alguma informação não seja identificada.
            - Escolher o formato correto conforme o contexto identificado (música, álbum ou nenhum dos dois).

        Returns:
            callable: Um pipeline composto por prompt, modelo de linguagem e parser de saída JSON.
        """
        class MusicDetails(BaseModel):
            artist: str
            track: str
            title: str
            
        llm = ChatGroq(model_name=self.model_name, temperature=self.temperature)
        parser = JsonOutputParser(pydantic_object=MusicDetails)        
        prompt = ChatPromptTemplate.from_messages([            
            ("system", '''You are an assistant whose task is to extract structured data in JSON format from input text.

            - Always respond with a single valid JSON object only, with no explanation.
            - If the text describes a **song**, extract:
                {{
                    "artist": "artist name",
                    "track": "track name",
                    "title": "artist name - track name"
                }}
            - Use "unknown" for any field not explicitly mentioned in the text.
            - If the text is about an **album** (and not about a single song), extract:
                {{
                    "artist": "artist name",
                    "album": "album name"
                }}
            - Use "unknown" for any missing field.
            - Do **not** include fields that were not requested in the chosen format.
            - If neither a song nor album can be identified, return a JSON object with all fields as "unknown".

            **Examples:**

            Input:  
            "The song 'Little Wing' by Jimi Hendrix is amazing."  
            Output:  
            {{
                "artist": "Jimi Hendrix",
                "track": "Little Wing",
                "title": "Jimi Hendrix - Little Wing"
            }}

            Input:  
            "The album 'Hybrid Theory' from Linkin Park defined a generation."  
            Output:  
            {{
                "artist": "Linkin Park",
                "album": "Hybrid Theory"
            }}

            Input:  
            "This is a music channel about heavy metal."  
            Output:  
            {{
                "artist": "unknown",
                "track": "unknown",
                "title": "unknown"
            }}'''), ("user", "{input}")
        ])
        
        return prompt | llm | parser

    def _get_spotify_access_token(self) -> str:
        """
        Solicita um novo access token do Spotify usando o refresh token.
        
        Returns:
            str: Token de acesso válido para uso nas APIs do Spotify.
        """
        url = 'https://accounts.spotify.com/api/token'
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        resp = requests.post(url, data=data)
        resp.raise_for_status()

        return resp.json().get('access_token', '')
    
    def _get_playlist_id(self) -> str:
        """
        Retorna o ID da playlist 'Youtube Scrapping' do usuário no Spotify.
        Se não existir, cria e retorna a nova playlist.
        """
        playlist_ep = f"{self.spotify_api}/users/{self.user_id}/playlists"
        user_playlists = requests.get(playlist_ep, headers=self.headers).json()

        for playlist in user_playlists['items']:
            if 'Youtube Scrapping' in playlist['name']:
                return playlist['id']

        payload = {
            'name': 'Youtube Scrapping',
            'description': 'Musicas que foram retiradas do Youtube',
            'public': True
        }
        return requests.post(playlist_ep, headers=self.headers, json=payload).json()['id']

    def _get_channel_id(self) -> str:
        """
        Busca e retorna o channel_id do canal a partir do nome.
        """
        url = (f"{self.youtube_api}/search?part=snippet&type=channel&q={self.channel_name}&key={self.yt_api_key}")
        resp = requests.get(url).json()
        return resp["items"][0]["snippet"]["channelId"]
    
    def _get_uploads_playlist_id(self, channel_id: str) -> str:
        """
        Busca e retorna o playlist_id dos uploads do canal.
        """
        url = (f"{self.youtube_api}/channels?part=contentDetails&id={channel_id}&key={self.yt_api_key}")
        resp = requests.get(url).json()
        return resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    
    def _get_all_playlist_videos(self, playlist_id: str) -> list[dict]:
        """
        Busca todos os vídeos da playlist, realizando paginação se necessário.
        """
        url = (f"{self.youtube_api}/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=50&key={self.yt_api_key}")
        videos = []
        while url:
            resp = requests.get(url).json()
            videos.extend(resp.get("items", []))
            next_page_token = resp.get("nextPageToken")
            if next_page_token:
                url = url + f"&pageToken={next_page_token}"
            else:
                url = None
        return videos
    
    def _over_days(self, published_date: str, days: int = 30) -> bool:
        """
        Verifica se a data publicada está acima de X dias atrás.

        Args:
            published_date (str): Data no formato ISO 8601 (ex: '2022-05-12T12:35:20Z')
            days (int): Quantidade de dias para diferença.

        Returns:
            bool: True se passou do valor de 'days', False caso contrário.
        """
        published_at = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - published_at).days > days
    
    def _is_short_video(self, item: dict) -> bool:
        """
        Retorna True se o vídeo for um 'Short' (duração menor que 1 minuto).
        """
        video_id = item["snippet"]["resourceId"]["videoId"]
        url = (f"{self.youtube_api}/videos?part=contentDetails&id={video_id}&key={self.yt_api_key}")
        resp = requests.get(url).json()
        duration = resp['items'][0]['contentDetails']['duration']

        return 'M' not in duration and 'H' not in duration
    
    def _filter_videos(self, videos: list[dict], days_limit: int) -> list[dict]:
        """
        Aplica filtros: por data e por ser short ou não.
        Retorna apenas metadados desejados.
        """
        result = []
        for item in videos:
            if self._over_days(item["snippet"]["publishedAt"], days_limit):
                break
            if not self._is_short_video(item):
                result.append({
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"]
                })
        return result

    def _get_youtube_links(self, description: str) -> list | str:
        """
        Extrai e retorna uma lista de links do YouTube da descrição.
        Se não houver links, retorna a própria descrição.
        """
        pattern = r"https:\/\/www\.youtube\.com\/watch\?v=[\w\-]+"
        links = re.findall(pattern, description)

        return links if links else description

    def _get_video_title(self, link: str) -> str:
        """
        Retorna o título de um vídeo do YouTube dado o link.
        """
        video_id = link.split("v=")[-1]
        url = (f"{self.youtube_api}/videos?part=snippet&id={video_id}&key={self.yt_api_key}")
        resp = requests.get(url).json()

        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["title"]
        return ""
    
    def _get_videos_metadata(self) -> list[str]:
        """
        Coleta metadados dos vídeos do canal conforme o modo (full ou limitando por dias).
        Retorna uma lista de títulos ou títulos+descrição caso não haja links na descrição.
        """
        days_limit = self.days
        channel_id = self._get_channel_id()
        playlist_id = self._get_uploads_playlist_id(channel_id)
        all_videos = self._get_all_playlist_videos(playlist_id)
        filtered_videos = self._filter_videos(all_videos, days_limit)

        videos_metadata = []
        for video in filtered_videos:
            links = self._get_youtube_links(video["description"])
            if isinstance(links, list):
                for link in links:
                    videos_metadata.append(self._get_video_title(link))
            else:
                videos_metadata.append(f"{video['title']}: {video['description']}")
        return videos_metadata

    def _get_wait_time(self, error: str) -> int:
        """
        Extrai o tempo de espera (em segundos) de uma mensagem de erro textual.
        Caso não seja possível extrair, retorna 120 segundos.
        """
        try:
            seconds = str(error).split('Please try again in ')[-1].split('s', 1)[0]
            seconds = float(seconds.replace('.', '').replace('m', '.'))
            return int(seconds * 60)
        except Exception:
            logging.warning("Falha ao extrair tempo de espera da mensagem de erro.")
            return 120

    def _ask(self, description: str) -> dict[str, str]:
        """
        Solicita extração dos campos musicais a partir de uma descrição textual usando o modelo LLM.
        Realiza até 3 tentativas automáticas em caso de falhas; após isso, retorna resposta padrão.
        """
        default_response = {"artist": "Unknown", "track": "Unknown", "title": "Unknown"}

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
        def _attempt() -> dict[str, str]:
            return self.chat.invoke(input=description)

        try:
            return _attempt()
        except OutputParserException:
            logging.warning(f"Falha ao converter resposta: {description}")
        except Exception as error:
            wait_time = self._get_wait_time(str(error))
            logging.error(f"Nova tentativa em {wait_time} segundos.")
            time.sleep(wait_time)
            try:
                return self.chat.invoke(input=description)
            except Exception as final_error:
                logging.error(f"Extração de dados falhou para: '{description}'. Erro: {final_error}")

        return default_response
    
    def _get_track_metadata(self, artist: str, track: str) -> Optional[Tuple[str, str, str]]:
        """
        Consulta a API do Spotify e retorna (uri, nome da faixa, nome do artista) da melhor correspondência,
        ou None caso não encontrado ou haja erro na requisição.
        """
        try:
            params = {
                "q": f"artist:{artist} track:{track}",
                "type": "track"
            }
            url = f'{self.spotify_api}/search'
            resp = requests.get(url, headers=self.headers, params=params)
            resp.raise_for_status()
            items = resp.json().get('tracks', {}).get('items', [])

            if not items:
                return None

            track_info = items[0]
            return track_info['uri'], track_info['name'], track_info['artists'][0]['name']

        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao localizar informações da faixa: {e}")
            return None
        
    def _get_existing_tracks(self) -> Set[Tuple[str, str]]:
        """
        Busca todas as faixas já presentes na playlist principal e retorna um conjunto
        de tuplas (nome_da_faixa, nome_do_artista) para evitar duplicatas.
        Em caso de erro, retorna um conjunto vazio.
        """
        try:
            response = requests.get(self.playlist_url, headers=self.headers)
            response.raise_for_status()
            playlist = response.json()['items']
            return {(item['track']['name'], item['track']['artists'][0]['name']) for item in playlist}
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao localizar faixas na playlist: {e}")
            return set()
        
    def _add_track_to_playlist(self, track_uri: str) -> bool:
        """
        Adiciona a faixa indicada (track_uri) à playlist principal do usuário no Spotify.
        Retorna True se a inclusão for bem-sucedida, False caso contrário.
        """
        try:
            resp = requests.post(
                self.playlist_url,
                headers=self.headers,
                json={'uris': [track_uri]}
            )
            resp.raise_for_status()
            return resp.status_code == 201
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao adicionar faixa à playlist: {e}")
            return False
        
    def _find_album_id(self, artist: str, album: str) -> str | None:
        """Busca o album_id do Spotify a partir de artista e álbum."""
        params = {"q": f"album:{album} artist:{artist}", "type": "album", "limit": 1}
        resp = requests.get(f"{self.spotify_api}/search", headers=self.headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = data["albums"]["items"]
        if not items:
            return None
        album_uri = items[0]["uri"]
        return album_uri.split(":")[-1]
    
    def _get_album_track_uris(self, album_id: str) -> list[str]:
        """Retorna a lista de uris das faixas do álbum (todas páginas)."""
        uris = []
        url = f"{self.spotify_api}/albums/{album_id}/tracks"
        params = {"limit": 50}
        while url:
            resp = requests.get(url, headers=self.headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            uris.extend(t["uri"] for t in data["items"])
            url = data.get("next")
            params = None
        return uris
    
    def _get_track_name_from_uri(self, uri: str) -> str:
        """
        Recupera o nome de uma faixa pelo seu Spotify URI.
        Idealmente, essa info já estaria disponível, mas se não, faça uma consulta.
        """
        track_id = uri.split(":")[-1]
        url = f"{self.spotify_api}/tracks/{track_id}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("name", "")

    def _add_tracks_to_playlist(self, track_uris: list[str], artist: str) -> None:
        """Adiciona as faixas à playlist (pulando duplicadas)."""
        existing_tracks = self._get_existing_tracks()
        for uri in track_uris:
            track_name = self._get_track_name_from_uri(uri)
            key = (track_name, artist)
            if key not in existing_tracks:
                if not self._add_track_to_playlist(uri):
                    logging.error(f"Falha ao adicionar faixa: {track_name} - {artist}.")
                    
    def _add_album_tracks(self, artist: str, album: str) -> None:
        """
        Busca o álbum do artista informado e adiciona todas as faixas à playlist principal do usuário.
        """
        album_id = self._find_album_id(artist, album)
        if not album_id:
            logging.warning(f"Álbum '{album}' de '{artist}' não encontrado no Spotify.")
            return

        track_uris = self._get_album_track_uris(album_id)
        self._add_tracks_to_playlist(track_uris, artist)
    
    def run(self) -> None:
        """
        Executa o pipeline principal:
        - Coleta metadados dos vídeos do canal.
        - Usa LLM para estruturar informações.
        - Separa músicas, álbuns e desconhecidos.
        - Adiciona faixas individuais e de álbuns à playlist, evitando duplicatas.
        """
        tracks, albums, unknowns = [], [], []
        for metadata in self._get_videos_metadata():
            track_info = self._ask(metadata)
            if track_info.get('album') and track_info['album'].lower() != "unknown":
                albums.append(track_info)
            elif track_info.get('title') and track_info['title'].lower() != 'unknown':
                tracks.append(track_info)
            else:
                unknowns.append(metadata)
        
        tracks_data = [
            self._get_track_metadata(track['artist'], track['track']) 
            for track in tracks
        ]
        tracks_data = [t for t in tracks_data if t]

        existing_tracks = self._get_existing_tracks()
        for uri, name, artist in tqdm(tracks_data, desc="Adicionando faixas únicas:", ncols=80):          
            if (name, artist) not in existing_tracks:
                if not self._add_track_to_playlist(uri):
                    logging.error(f"Falha ao adicionar faixa: {name} - {artist}.")

        for album in tqdm(albums, desc="Adicionando álbuns completos:", ncols=80):
            self._add_album_tracks(album['artist'], album['album'])

        logging.info('Execução finalizada!')

# %%
if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()
    
    app = App(
        channel_url=args.url,
        full=args.full,
        model_name=args.model_name,
        temperature=args.temperature
    )
    
    app.run()
