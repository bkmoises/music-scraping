import os
import logging
import requests
from datetime import datetime, timezone, timedelta

def get_access_token(refresh_token, client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()['access_token']

def get_playlist_id(name="Youtube Scrapping"):
    playlist_url = f"{SPOTIFY_API}/users/{USER_ID}/playlists?limit=50"
    resp = requests.get(playlist_url, headers=HEADERS)
    resp.raise_for_status()
    playlists = resp.json().get("items", [])
    while resp.json().get("next"):
        resp = requests.get(resp.json()["next"], headers=HEADERS)
        playlists += resp.json().get("items", [])
    for playlist in playlists:
        if name in playlist["name"]:
            return playlist["id"]
    raise ValueError("Playlist não encontrada.")

def get_all_playlist_tracks(playlist_id):
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/tracks?limit=100"
    all_items = []
    while url:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        all_items.extend(items)
        url = data.get("next")
    return all_items

def remove_tracks_by_position(playlist_id, track_positions):
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/tracks"
    payload = {"tracks": track_positions}
    resp = requests.delete(url, headers=HEADERS, json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Erro ao remover faixas: {resp.status_code} - {resp.text}")
        raise e
    logging.info(f"Removidas {len(track_positions)} ocorrências da playlist {playlist_id}")

def remove_tracks_from_scrapping_last_30_days():
    playlist_id = get_playlist_id()
    all_items = get_all_playlist_tracks(playlist_id)
    now = datetime.now(timezone.utc)
    production_limit = now - timedelta(days=30)
    removals = {}

    for idx, item in enumerate(all_items):
        added_at = item.get("added_at")
        track = item.get("track")
        if not (added_at and track and track.get("uri")):
            continue
        try:
            added_time = datetime.strptime(added_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception as e:
            logging.warning(f"Data de adição inválida: {added_at} ({e})")
            continue
        if added_time < production_limit:
            uri = track["uri"]
            if uri not in removals:
                removals[uri] = []
            removals[uri].append(idx)

    tracks_to_remove = [{"uri": uri, "positions": positions} for uri, positions in removals.items()]
    if tracks_to_remove:
        remove_tracks_by_position(playlist_id, tracks_to_remove)
    else:
        logging.info("Nenhuma faixa superior a 30 dias para remover.")

if __name__ == "__main__":
    USER_ID       = os.environ.get("USER_ID")
    CLIENT_ID     = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")
    
    SPOTIFY_API = "https://api.spotify.com/v1"
    ACCESS_TOKEN = get_access_token(REFRESH_TOKEN, CLIENT_ID, CLIENT_SECRET)
    HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    
    remove_tracks_from_scrapping_last_30_days()
