# %%
import os
import spotify
import logging
from datetime import datetime

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
def main(user_id: str, refresh_token: str, client_id: str, client_secret: str):
    """ 
    Função principal para seguir artistas de uma playlist específica do Spotify.
    
    Inputs:
    - user_id (str): ID do usuário do Spotify.
    - refresh_token (str): Token de atualização para autenticação na API do Spotify.
    - client_id (str): ID do cliente da aplicação Spotify.
    - client_secret (str): Segredo do cliente da aplicação Spotify.
    """
    playlist_name  = f"Box {datetime.now().year}"

    token    = spotify.get_token(refresh_token, client_id, client_secret)
    playlist = spotify.get_playlist(user_id, playlist_name, token)
    tracks   = spotify.get_playlist_tracks(playlist.id, token)
    artists  = spotify.get_followed_artists(token)

    following_list = [artist.id for artist in artists]

    for track in tracks:
        if track.artist.id not in following_list:
            spotify.follow_artist(track, token)

# %%
if __name__ == "__main__":
    USER_ID       = os.environ.get("USER_ID")
    CLIENT_ID     = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")
    
    main(
        user_id=USER_ID,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
