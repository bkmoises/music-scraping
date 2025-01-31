# %%
import os
import json
import time
import socket
import logging
import requests
import threading
import webbrowser
import urllib.parse

from dotenv import load_dotenv
from urllib.parse import urlparse
from typing import Dict, Optional

from http.server import BaseHTTPRequestHandler, HTTPServer

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            logging.warning(f"Arquivo não encontrado. {file_name}")
            return {}

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao fazer a requisição: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao decodificar o JSON: {e}")
    except KeyError as e:
        logging.error(f"Erro: Chave ausente na resposta: {e}")

    return None

# %%
def get_auth_code(client_id: str, redirect_uri: str) -> str:
    url = 'https://accounts.spotify.com/authorize'
    port = urlparse(redirect_uri).port
    params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
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

# %%
def get_access_token(code: str, redirect_uri: str, client_id: str, client_secret: str) -> str:
    url = 'https://accounts.spotify.com/api/token'

    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri,
        'client_id': client_id,
        'client_secret': client_secret,
    }

    return requests.post(url, data=data).json().get('access_token')

# %%
load_dotenv()

GIST_ID = os.getenv('GIST_ID')
GIST_TOKEN = os.getenv('GIST_TOKEN')
GIST_ID_DATABASE = os.getenv('GIST_ID_DATABASE')
GIST_FILE = "youtube-music-scrapping.json"
GIST_FILE_DATABASE = 'spotify-backup.json'

database = get_file_from_gist(GIST_ID_DATABASE, GIST_TOKEN, GIST_FILE_DATABASE)
credentials = get_file_from_gist(GIST_ID, GIST_TOKEN, GIST_FILE)

USER_ID = credentials['user_id']
CLIENT_ID = credentials['client_id']
CLIENT_SECRET= credentials['client_secret']
REDIRECT_URI = credentials['redirect_uri']

CODE = get_auth_code(CLIENT_ID, REDIRECT_URI)
ACCESS_TOKEN = get_access_token(CODE, REDIRECT_URI, CLIENT_ID, CLIENT_SECRET)

# %%
if __name__ == '__main__':
    spotify_api = 'https://api.spotify.com/v1'
    playlist_url = f'{spotify_api}/users/{USER_ID}/playlists'
    gist_url = f"https://api.github.com/gists/{GIST_ID_DATABASE}"
    
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

    user_playlists = requests.get(playlist_url, headers=headers).json()

    playlists = [{"id": playlist['id'], "name": playlist['name']} for playlist in user_playlists['items']]
    
    for playlist in playlists:
        url = f'{spotify_api}/playlists/{playlist['id']}/tracks'
        
        playlist_tracks = requests.get(url, headers=headers).json()
        tracks = [{'track_id': track['track']['id'], 'artist': track['track']['artists'][0]['name'], 'name': track['track']['name']} for track in playlist_tracks['items']]

        playlist['tracks'] = tracks
        
    content = json.dumps(playlists, indent=2)
    description = f'Backup realizado em {time.strftime("%d/%m/%Y as %H:%M:%S", time.localtime())}'
    headers = {"Authorization": f"token {GIST_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    data = {"description": description, "files": {GIST_FILE_DATABASE: {"content": content}}}

    response = requests.patch(gist_url, headers=headers, json=data)

    if response.status_code == 200:
        logging.info("Backup realizado com sucesso!")
    else:
        logging.error(f"Falha ao realizar o backup: {response.status_code}, {response.text}")


