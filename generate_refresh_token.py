import base64
import requests

CLIENT_ID = ""
REDIRECT_URI = "http://127.0.0.1:8888/callback"
CLIENT_SECRET = ""

url = f"https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=user-follow-read+user-follow-modify+user-follow-read+user-follow-modify+playlist-modify-public+playlist-modify-private+user-library-read+user-library-modify"

auth_code = "" # resgatar do link
token_url = "https://accounts.spotify.com/api/token"

credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
credentials_b64 = base64.b64encode(credentials.encode()).decode()

headers = {
    "Authorization": f"Basic {credentials_b64}",
    "Content-Type": "application/x-www-form-urlencoded"
}

data = {
    "grant_type": "authorization_code",
    "code": auth_code,
    "redirect_uri": REDIRECT_URI
}

response = requests.post(token_url, headers=headers, data=data)

print(f"Status: {response.status_code}")
print(f"Resposta: {response.text}")

if response.status_code == 200:
    refresh_token = response.json()["refresh_token"]
    print(f"\n✅ Refresh token: {refresh_token}")
else:
    print(f"❌ Erro: {response.text}")