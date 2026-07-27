# %%
import os
import logging
import spotify
from datetime import datetime, timezone, timedelta

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# %%
def main(user_id: str, refresh_token: str, client_id: str, client_secret: str):
    """
    Função principal que remove faixas antigas de uma playlist do Spotify.
    
    Inputs:
    - user_id (str): ID do usuário do Spotify.
    - refresh_token (str): Token de atualização para autenticação na API do Spotify.
    - client_id (str): ID do cliente da aplicação Spotify.
    - client_secret (str): Segredo do cliente da aplicação Spotify.
    """
    now   = datetime.now(timezone.utc)
    limit = now - timedelta(days=30)

    token    = spotify.get_token(refresh_token, client_id, client_secret)
    playlist = spotify.get_playlist(user_id, "New Rock Hits", token)
    tracks   = spotify.get_playlist_tracks(playlist.id, token)

    tracks_to_remove = []

    for track in tracks:
        added_at = track.added_at
        if added_at:
            added_at_dt = datetime.fromisoformat(added_at.replace("Z", "+00:00"))
            if added_at_dt < limit:
                tracks_to_remove.append({"uri": track.uri})
                logging.info(f"Removendo faixa: {track.name} - {track.artist.name} adicionada em {added_at_dt.isoformat()}")
                
    if tracks_to_remove:
        spotify.remove_tracks(playlist.id, tracks_to_remove, token)
        return 0

    logging.info("Nenhuma faixa antiga encontrada para remoção.")

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
