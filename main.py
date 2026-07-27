# %%
import os
import logging
import argparse
import spotify
import youtube

from model import Model
from dotenv import load_dotenv

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
parser = argparse.ArgumentParser(description='Scrape music data from a YouTube channel')
parser.add_argument('--days_back', type=int, default=1, help='Number of days back to scrape chat data from')
parser.add_argument('--model_name', type=str, help='LLM model name to use for processing the chat data')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')
parser.add_argument('--playlist_name', type=str, default="New Rock Hits", help='Spotify playlist name to add tracks to')

# %%
def remove_duplicate_tracks(track_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Remove duplicatas normalizando artist + track em lowercase.
    Mantém a primeira ocorrência.
    
    Inputs:
    - track_list (list[dict]): Lista de dicionários com tracks.
    
    Returns:
    - list[dict]: Lista de musicas únicas.
    - list[dict]: Lista de álbuns únicos.
    """
    seen = set()
    unique_tracks = []
    unique_albums = []
    
    if any("track" in track for track in track_list):
        for track in track_list:
            if "track" not in track:
                continue
            track_key = f"{track['artist'].lower()} - {track['track'].lower()}"
            
            if track_key not in seen:
                seen.add(track_key)
                unique_tracks.append(track)

    if any("album" in track for track in track_list):
        for track in track_list:
            if "album" not in track:
                continue
            track_key = f"{track['artist'].lower()} - {track['album'].lower()}"
            
            if track_key not in seen:
                seen.add(track_key)
                unique_albums.append(track)
    
    return unique_tracks, unique_albums

# %%
def app(days_back: int, playlist_name: str, model_name: str, temperature: float):
    client_id = os.environ.get("CLIENT_ID")
    user_id = os.environ.get("USER_ID")
    client_secret = os.environ.get("CLIENT_SECRET")
    refresh_token = os.environ.get("REFRESH_TOKEN")
    yt_api_key = os.environ.get("YT_API_KEY")
    
    with open("channels.txt", "r") as f:
        channel_list = [line.strip() for line in f.readlines() if line.strip()]

# %%
    llm = Model(model_name=model_name, temperature=temperature)

    videos = []
    for channel_name in channel_list:
        videos += youtube.process(channel_name=channel_name, days_back=days_back, api_key=yt_api_key)
        
    tracks = []
    for video in videos:
        track_info = llm.ask(video.title)

        if track_info["artist"] == "unknown":
            track_info = llm.ask(video.description)
        if track_info["artist"] != "unknown":
            tracks.append(track_info)
            
    unique_tracks, unique_albums = remove_duplicate_tracks(tracks)

# %%
    spotify_token   = spotify.get_token(refresh_token, client_id, client_secret)
    playlist        = spotify.get_playlist(user_id, playlist_name, spotify_token)
    existing_tracks = spotify.get_playlist_tracks(playlist.id, spotify_token)

# %%
    album_ids = set()
    for ua in unique_albums:
        title, artist = ua["album"], ua["artist"]
        album = spotify.get_album(artist, title, spotify_token)
        
        if album and album.id not in album_ids:
            logging.info(f"Processando album: {title} - {artist}")
            album_ids.add(album.id)

# %%
    tracks_to_add = []
    for id in album_ids:
        album_tracks = spotify.get_album_tracks(id, spotify_token)
        tracks_to_add.extend(album_tracks)

# %%
    for ut in unique_tracks:
        artist, title = ut["artist"], ut["track"]
        track = spotify.get_track(artist, title, spotify_token)
        if track:
            logging.info(f"Processando track: {title} - {artist}")
            tracks_to_add.append(track)

# %%
    existing_uris = [track.uri for track in existing_tracks]
    track_uris    = [track.uri for track in tracks_to_add if track.uri not in existing_uris]

# %%
    spotify.add_tracks(playlist.id, track_uris, spotify_token)

# %%
if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()
    
    app(
        days_back=args.days_back, 
        playlist_name=args.playlist_name,
        model_name=args.model_name, 
        temperature=args.temperature
    )
