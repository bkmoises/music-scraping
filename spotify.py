# %%
import logging
import requests
from dataclasses import dataclass

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
@dataclass
class Track:
    uri: str
    artist: str
    name: str

# %%
SPOTIFY_API = "https://api.spotify.com/v1"

# %%
def get_token(refresh_token: str, client_id: str, client_secret: str) -> str:
    """
    Solicita um novo access token do Spotify usando o refresh token.
    
    Inputs:
    - refresh_token (str): Token de atualização obtido durante a autenticação inicial.
    - client_id (str): ID do cliente da aplicação Spotify.
    - client_secret (str): Segredo do cliente da aplicação Spotify.
    
    Returns:
    - str: Token de acesso válido para uso nas APIs do Spotify.
    """
    url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret
    }
    
    resp = requests.post(url, data=data)
    resp.raise_for_status()

    return resp.json().get('access_token', '')

# %%
def get_playlist_uri(user_id: str, token: str) -> str:
    """
    Retorna o id da playlist do usuário no Spotify.
    Se não existir, cria e retorna a nova playlist.
    
    Inputs:
    - user_id (str): id do usuário no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - str: id da playlist.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SPOTIFY_API}/users/{user_id}/playlists"

    playlists = requests.get(url, headers=headers).json()

    for playlist in playlists['items']:
        if 'New Rock Hits' in playlist['name']:
            return playlist['id']

    payload = {
        'name': 'New Rock Hits',
        'description': 'Os melhores lançamentos de rock atualizados todos os dias. Fique por dentro das novidades, dos hits que estão em alta e da melhor música que o rock está produzindo. Hard rock, metal, indie, punk e muito mais. Atualizado diariamente para você não perder nada.',
        'public': True
    }
    return requests.post(url, headers=headers, json=payload).json()['id']

# %%
def get_playlist_tracks(playlist_uri: str, token: str) -> set[str]:
    """
    Busca TODAS as faixas já presentes na playlist (com paginação)
    e retorna um conjunto de track_ids.
    
    Inputs:
    - playlist_uri (str): id da playlist no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - set[str]: Conjunto de track_ids já presentes na playlist.
    """
    url = f"{SPOTIFY_API}/playlists/{playlist_uri}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    track_ids: set[str] = set()
    
    try:
        params = {"limit": 100, "offset": 0}

        while url:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                track = item.get("track") or {}
                track_id = track.get("id")
                if track_id:
                    track_ids.add(track_id)

            url = data.get("next")
            params = None

        return track_ids
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar faixas existentes na playlist: {e}")
        return track_ids

# %%
def get_track_uri(artist: str, track: str, token: str) -> Track:
    """
    Consulta a API do Spotify e retorna um objeto Track da melhor correspondência,
    
    Inputs:
    - artist (str): Nome do artista.
    - track (str): Nome da faixa.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
        
    Returns: 
    - track (Track): objeto contendo a uri, nome da faixa e nome do artista.
    """
    url = f'{SPOTIFY_API}/search'
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        params = {
            "q": f"artist:{artist} track:{track}",
            "type": "track"
        }
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        items = resp.json().get('tracks', {}).get('items', [])

        if not items:
            return Track(uri='', name='', artist='')

        track_info = items[0]
        return Track(
            uri=track_info['uri'],
            name=track_info['name'],
            artist=track_info['artists'][0]['name']
        )

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao localizar informações da faixa: {e}")
        return Track(uri='', name='', artist='')

# %%
def get_album_uri(artist: str, album: str, token: str) -> str:
    """
    Busca o album_id do Spotify a partir de artista e álbum.

    Inputs:
    - artist (str): Nome do artista.
    - album (str): Nome do álbum.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.

    Returns:
    - str: album_id do Spotify ou string vazia se não encontrado.
    """
    params = {"q": f"album:{album} artist:{artist}", "type": "album", "limit": 1}

    url = f"{SPOTIFY_API}/search"
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    items = data["albums"]["items"]

    return '' if not items else items[0]["uri"].split(":")[-1]

# %%
def get_album_tracks(album_uri: str, token: str) -> list[Track]:
    """
    Retorna a lista de Track das faixas do álbum (todas páginas).

    Inputs:
    - album_uri (str): id do álbum no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - list[Track]: Lista de objetos Track representando as faixas do álbum.
    """
    params = {"limit": 50}
    url = f"{SPOTIFY_API}/albums/{album_uri}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    tracks = []
    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        tracks.extend(
            [Track(
                uri=t["uri"], 
                name=t["name"], 
                artist=t['artists'][0]['name']
            ) for t in data["items"]]
        )

        url = data.get("next")
        params = None

    return tracks

# %%
def add_tracks(playlist_uri: str, track_uris: list[str], token: str) -> dict:
    """
    Adiciona múltiplas faixas à playlist principal do usuário no Spotify.

    Inputs:
    - playlist_uri (str): URI da playlist onde as faixas serão adicionadas
    - track_uris (list[str]): Lista de URIs das faixas a adicionar
    - token (str): Token de autenticação do Spotify

    Returns:
    - dict: estatísticas {
        'total': int,
        'added': int,
        'failed': int,
        'errors': list[str]
    }
    """
    headers = {"Authorization": f"Bearer {token}"}
    playlist_url = f'{SPOTIFY_API}/playlists/{playlist_uri}/tracks'
    
    BATCH_SIZE = 100
    stats = {'total': len(track_uris), 'added': 0, 'failed': 0, 'errors': []}
    
    for i in range(0, len(track_uris), BATCH_SIZE):
        batch = track_uris[i:i + BATCH_SIZE]
        
        try:
            resp = requests.post(
                playlist_url,
                headers=headers,
                json={'uris': batch}
            )
            resp.raise_for_status()
            
            if resp.status_code == 201:
                stats['added'] += len(batch)
                logging.info(f"✓ {len(batch)} faixas adicionadas (lote {i//BATCH_SIZE + 1})")
            else:
                stats['failed'] += len(batch)
                stats["failed_uris"] = batch
                error_msg = f"Status inesperado {resp.status_code} no lote {i//BATCH_SIZE + 1}"
                stats['errors'].append(error_msg)
                logging.warning(error_msg)
                
        except requests.exceptions.RequestException as e:
            stats['failed'] += len(batch)
            stats["failed_uris"] = batch
            error_msg = f"Erro ao adicionar lote {i//BATCH_SIZE + 1}: {str(e)}"
            stats['errors'].append(error_msg)
            logging.error(error_msg)
    
    logging.info(f"Resumo: {stats['added']}/{stats['total']} faixas adicionadas")
    return stats

# %%
def get_track_data(uri: str, token: str) -> Track:
    """
    Recupera as informações de uma faixa a partir de sua URI no Spotify.
    
    Inputs:
    - uri (str): URI da faixa no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
        
    Returns:
    - Track: Objeto contendo a URI, nome da faixa e nome do artista.
    """
    track_id = uri.split(":")[-1]
    headers = {"Authorization": f"Bearer {token}"}
    
    url = f"{SPOTIFY_API}/tracks/{track_id}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    track_info = resp.json()
    
    return Track(
        uri=uri,
        name=track_info["name"],
        artist=track_info['artists'][0]['name']
    )
