# %%
import logging
import requests
from dataclasses import dataclass

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
SPOTIFY_API = "https://api.spotify.com/v1"

# %%
@dataclass
class Album:
    id: str
    name: str
    release_date: str
    uri: str
    href: str
    
    @classmethod
    def save(cls, data: dict) -> "Album":
        """Factory method para criar a partir da resposta do Spotify"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            release_date=data.get("release_date"),
            uri=data.get("uri"),
            href=data.get("href")
        )
        
@dataclass
class Artist:
    id: str
    name: str
    uri: str
    
    @classmethod
    def save(cls, data: dict) -> "Artist":
        """Factory method para criar a partir da resposta do Spotify"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            uri=data.get("uri")
        )

@dataclass
class Track:
    id: str
    name: str
    uri: str
    artist: Artist
    album: Album
    added_at: str
    
    @classmethod
    def save(cls, data: dict, added_at: str = "") -> "Track":
        """Factory method para criar a partir da resposta do Spotify"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            uri=data.get("uri"),
            artist=Artist.save(data.get("artists", [{}])[0]),
            album=Album.save(data.get("album", {})),
            added_at=added_at
        )
        
@dataclass
class Playlist:
    id: str
    name: str
    uri: str
    url: str
    count: int
    
    @classmethod
    def save(cls, data: dict) -> "Playlist":
        """Factory method para criar a partir da resposta do Spotify"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            uri=f"spotify:playlist:{data.get('id')}",
            url=data.get("tracks", {}).get("href"),
            count=data.get("tracks", {}).get("total")
        )

# %%
def get_token(refresh_token: str, client_id: str, client_secret: str) -> str:
    """
    Solicita um novo access token do Spotify usando o refresh token.
    
    Args:
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
def get_playlist(user_id: str, playlist_name: str, token: str) -> Playlist | None:
    """
    Busca uma playlist pelo nome dela.
    Retorna um objeto Playlist se encontrada, caso contrário retorna None.
    
    Args:
    - user_id (str): ID do usuário do Spotify.
    - playlist_name (str): Nome da playlist a ser buscada.
    - token (str): Token de acesso para a API do Spotify.
    
    Returns:
    - Playlist | None: Objeto Playlist se encontrada, caso contrário None.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SPOTIFY_API}/users/{user_id}/playlists"
    
    playlists = requests.get(url, headers=headers).json()

    for playlist in playlists['items']:
        if playlist_name in playlist['name']:
            return Playlist.save(playlist)
    
    return None

# %%
def create_playlist(user_id: str, playlist_name: str, description: str, public: bool, token: str) -> Playlist:
    """
    Cria uma nova playlist para o usuário especificado.
    
    Args:
    - user_id (str): id do usuário no Spotify.
    - playlist_name (str): Nome da playlist a ser criada.
    - description (str): Descrição da playlist.
    - public (bool): Define se a playlist será pública.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - Playlist: objeto da playlist criada.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SPOTIFY_API}/users/{user_id}/playlists"

    payload = {
        'name': playlist_name,
        'description': description,
        'public': public
    }

    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    playlist = res.json()

    return Playlist.save(playlist)

# %%
def get_playlist_tracks(playlist_id: str, token: str) -> list[Track]:
    """
    Busca TODAS as faixas já presentes na playlist (com paginação)
    e retorna uma lista de Track.
    
    Args:
    - playlist_id (str): id da playlist no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - list[Track]: Lista de faixas já presentes na playlist.
    """
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    tracks: list[Track] = []
    
    try:
        params = {"limit": 100}
        while url:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                track = item.get("track") or {}
                track_id = track.get("id")
                if track_id:
                    tracks.append(Track.save(track, added_at=item.get("added_at", "")))

            url = data.get("next")
            params = None

        return tracks
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar faixas existentes na playlist: {e}")
        return tracks

# %%
def get_track(artist: str, track: str, token: str) -> Track | None:
    """
    Consulta a API do Spotify e retorna um objeto Track da melhor correspondência,
    
    Args:
    - artist (str): Nome do artista.
    - track (str): Nome da faixa.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
        
    Returns: 
    - track (Track) | None: objeto contendo a uri, nome da faixa e nome do artista, ou None se não encontrado.
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
            return None

        track_info = items[0]
        return Track.save(track_info)

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao localizar informações da faixa: {e}")
        return None

# %%
def get_album(artist: str, album: str, token: str) -> Album | None:
    """
    Busca o album_id do Spotify a partir de artista e álbum.

    Args:
    - artist (str): Nome do artista.
    - album (str): Nome do álbum.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.

    Returns:
    - Album: objeto Album do Spotify ou None se não encontrado.
    """
    params = {"q": f"album:{album} artist:{artist}", "type": "album", "limit": 1}

    url = f"{SPOTIFY_API}/search"
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    items = data["albums"]["items"]

    return Album.save(items[0]) if items else None

# %%
def get_album_tracks(album_id: str, token: str) -> list[Track]:
    """
    Retorna a lista de Track das faixas do álbum (todas páginas).

    Args:
    - album_id (str): id do álbum no Spotify.
    - token (str): Token de acesso válido para uso nas APIs do Spotify.
    
    Returns:
    - list[Track]: Lista de objetos Track representando as faixas do álbum.
    """
    params = {"limit": 50}
    url = f"{SPOTIFY_API}/albums/{album_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    tracks = []
    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        tracks.extend(Track.save(track) for track in data.get("items", []))

        url = data.get("next")
        params = None

    return tracks

# %%
def add_tracks(playlist_id: str, track_uris: list[str], token: str) -> dict:
    """
    Adiciona múltiplas faixas à playlist principal do usuário no Spotify.

    Args:
    - playlist_id (str): ID da playlist onde as faixas serão adicionadas
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
    playlist_url = f'{SPOTIFY_API}/playlists/{playlist_id}/tracks'
    
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
def remove_tracks(playlist_id: str, tracks_to_remove: list[dict], token: str):
    """
    Remove faixas de uma playlist do Spotify.

    Args:
    - playlist_id (str): O URI da playlist do Spotify.
    - tracks_to_remove (list[dict]): A lista de dicionários de faixas a serem removidas.
    - token (str): O token de acesso válido para uso nas APIs do Spotify.
    """
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/items"
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = requests.delete(url, headers=headers, json={"items": tracks_to_remove})

    try:
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao remover faixas: {resp.status_code} - {resp.text}")
        raise e

    logging.info(f"Removidas {len(tracks_to_remove)} faixas da playlist")

# %%
def follow_artist(track: Track, token: str) -> None:
    """
    Segue o artista da faixa especificada no Spotify.
    
    Args:
    - track (Track): Objeto Track representando a faixa no Spotify.
    - token (str): Token de acesso válido para a API do Spotify.
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        follow_response = requests.put(
            f"{SPOTIFY_API}/me/following",
            params={"type": "artist", "ids": track.artist.id},
            headers=headers
        )
        follow_response.raise_for_status()
        
        logging.info(f"Seguindo artista: {track.artist.name} (ID: {track.artist.id})")
    
    except Exception as e:
        logging.info(f"Erro ao seguir artista {track.artist.name} (ID: {track.artist.id}): {str(e)}")

# %%
def get_followed_artists(token: str) -> list[Artist]:
    """
    Retorna uma lista com os IDs de todos os artistas que você segue no Spotify.
    
    Args:
    - token (str): Token de acesso para autenticação na API do Spotify.
    
    Returns:
    - list[Artist]: Lista de objetos Artist dos artistas seguidos.
    """
    url = f"{SPOTIFY_API}/me/following"
    headers = {"Authorization": f"Bearer {token}"}
    
    artists_list = []
    after = None
    
    while True:
        params = {"type": "artist", "limit": 50}
        if after:
            params["after"] = after
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        artists = data["artists"]["items"]
        
        artists_list.extend([Artist.save(artist) for artist in artists])
        
        if data["artists"]["next"]:
            after = data["artists"]["cursors"]["after"]
        else:
            break
    return artists_list


