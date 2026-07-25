# %%
import re
import logging
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
@dataclass
class Video:
    id: str
    title: str
    description: str

# %%
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

# %%
def get_channel_id(channel_name: str, api_key: str) -> str:
    """
    Busca e retorna o channel_id do canal a partir do nome.
    
    Inputs:
    - channel_name (str): Nome do canal a ser buscado.
    - api_key (str): Chave de API do YouTube.

    Returns:
    - channel_id (str): ID do canal correspondente ao nome fornecido.
    """
    url = (f"{YOUTUBE_API}/search?part=snippet&type=channel&q={channel_name}&key={api_key}")
    resp = requests.get(url).json()

    return resp["items"][0]["snippet"]["channelId"]

# %%
def get_playlist_id(channel_id: str, api_key: str) -> str:
    """
    Busca e retorna o playlist_id dos uploads do canal.

    Inputs:
    - channel_id (str): ID do canal.
    - api_key (str): Chave de API do YouTube.

    Returns:
    - playlist_id (str): ID da playlist de uploads do canal.
    """
    url = (f"{YOUTUBE_API}/channels?part=contentDetails&id={channel_id}&key={api_key}")
    resp = requests.get(url).json()

    return resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

# %%
def get_playlist_videos(playlist_id: str, api_key: str, days_back: int = 1) -> list[Video]:
    """
    Busca todos os vídeos da playlist, com opção de filtro por período de tempo.
    
    Inputs:
    - playlist_id (str): ID da playlist.
    - api_key (str): Chave de API do YouTube.
    - days_back (int, optional): Número de dias para filtrar retroativamente.
                                Se 0, retorna todos os vídeos.
    
    Returns:
    - videos (list[Video]): Lista de objetos Video contendo informações dos vídeos 
                            publicados no período especificado.
    """
    url = f"{YOUTUBE_API}/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=50&key={api_key}"

    videos = []
    
    date_limit = None
    if days_back > 0:
        date_limit = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    while url:
        resp = requests.get(url).json()
        items = resp.get("items", [])
        
        if date_limit:
            for item in items:
                published_at = item.get("snippet", {}).get("publishedAt")
                if published_at:
                    published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    
                    if published_date >= date_limit:
                        item_meta = item.get("snippet", {})
                        
                        videos.append(Video(
                            id=item_meta.get("resourceId", {}).get("videoId"),
                            title=item_meta.get("title"),
                            description=item_meta.get("description")
                        ))
                    else:
                        return videos
        else:
            videos.extend([Video(
                id=item.get("snippet", {}).get("resourceId", {}).get("videoId"),
                title=item.get("snippet", {}).get("title"),
                description=item.get("snippet", {}).get("description")
            ) for item in items])
        
        next_page_token = resp.get("nextPageToken")
        if next_page_token:
            url = url + f"&pageToken={next_page_token}"
        else:
            url = None
    
    return videos

# %%
def get_links(description: str) -> list[str] | str:
    """
    Extrai e retorna uma lista de links do YouTube da descrição.
    Se não houver links, retorna a própria descrição.
    
    Inputs:
    - description (str): Descrição do vídeo.
    
    Returns:
    - links (list[str] | str): Lista de links encontrados na descrição ou 
                                a própria descrição se nenhum link for encontrado.
    """
    pattern = r"https:\/\/www\.youtube\.com\/watch\?v=[\w\-]+"
    links = re.findall(pattern, description)

    return links if links else description

# %%
def get_video(link: str, api_key: str) -> Video:
    """
    Retorna um objeto Video do YouTube dado o link.
    """
    video_id = link.split("v=")[-1]
    url = f"{YOUTUBE_API}/videos?part=snippet&id={video_id}&key={api_key}"

    resp = requests.get(url).json()

    items = resp.get("items", [])
    if items:
        snippet = items[0]["snippet"]
        return Video(
            id=video_id,
            title=snippet["title"],
            description=snippet["description"]
        )
    return Video(id="", title="", description="")

# %%
def process(channel_name: str, days_back: int, api_key: str) -> list[Video]:
    """
    Processa o canal e retorna uma lista de vídeos publicados nos últimos 'days_back' dias.
    
    Inputs:
    - channel_name (str): Nome do canal.
    - days_back (int): Número de dias para filtrar retroativamente.
    - api_key (str): Chave de API do YouTube.
    
    Returns:
    - videos (list[Video]): Lista de objetos Video publicados no período especificado.
    """
    channel_id  = get_channel_id(channel_name, api_key)
    playlist_id = get_playlist_id(channel_id, api_key)
    videos      = get_playlist_videos(playlist_id, api_key, days_back)

    track_datas = []
    processed_track_ids = set()

    for video in videos:
        video_link = get_links(video.description)
        if isinstance(video_link, list) and video_link:
            for link in video_link:
                track = get_video(link, api_key)

                if track.id not in processed_track_ids:
                    track_datas.append(track)
                    processed_track_ids.add(track.id)
        else:
            track_datas.append(video)
            
    return track_datas
