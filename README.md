# YouTube Music Scraping

Projeto para extração estruturada de dados musicais (faixas e álbuns) de canais do YouTube, com integração automática à playlist do Spotify do usuário, usando inteligência artificial para automação da classificação das músicas.

## O que o projeto faz?

- Scrapeia informações de vídeos de um canal do YouTube
- Usa IA (Large Language Model) via [LangChain](https://python.langchain.com/) para identificar se o vídeo menciona música, álbum ou outro
- Extrai metadados (artista, faixa, álbum)
- Busca e adiciona faixas e álbuns na playlist "Youtube Scrapping" do usuário no Spotify, evitando duplicatas

## Funcionalidades principais

- Coleta vídeos recentes (últimos 30 dias por padrão) ou todos os vídeos do canal
- Processamento automático de descrições/textos dos vídeos usando IA
- Busca dados na API do Spotify
- Faz interface segura usando variáveis de ambiente para credenciais sensíveis
- Tolerante a falhas com tentativas automáticas

## Como usar

### Pré-requisitos

- Python 3.10+
- Uma API Key do YouTube
- Credenciais OAuth e tokens do Spotify
- Modelos e API key da Groq ou outro LLM compatível
- Variáveis de ambiente bem configuradas (`CLIENT_ID`, `USER_ID`, `CLIENT_SECRET`, `REFRESH_TOKEN`, `YT_API_KEY`, ...)

### Instalação

```bash
git clone https://github.com/bkmoises/music-scraping.git
cd music-scraping
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz do projeto:

```
CLIENT_ID=seu_client_id_spotify
USER_ID=seu_user_id_spotify
CLIENT_SECRET=sua_client_secret_spotify
REFRESH_TOKEN=seu_refresh_token_spotify
YT_API_KEY=sua_api_key_do_youtube
GROQ_API_KEY=sua_api_key_do_groq
```

### Uso

Execute o script informando os argumentos necessários:

```bash
python youtube_data_scraper.py \
  --url "https://www.youtube.com/@SEUCANAL" \
  --model_name "groq-llama3-8b-8192" # Exemplo de modelo IA suportado
```

Argumentos principais:

- `--url`: URL do canal do YouTube (obrigatório)
- `--full`: Defina `True` para buscar todos os vídeos do canal, `False` (padrão) para só buscar os últimos 30 dias
- `--model_name`: Nome do modelo de IA a ser usado (obrigatório)
- `--temperature`: Parâmetro de criatividade do modelo (padrão 0.7)

Exemplo:

```bash
python youtube_data_scraper.py --url "https://www.youtube.com/@NOME" --model_name "groq-llama3-8b-8192"
```

## Exemplo de fluxo

1. Busca todos os vídeos do canal escolhido
2. Coleta títulos e descrições
3. IA extrai artista/música/álbum dos textos
4. Consulta a API do Spotify pelos dados extraídos
5. Adiciona automaticamente músicas e álbuns correspondentes na playlist
6. Evita duplicatas e realiza tentativas em caso de erro nas APIs

## Observações

- O projeto não baixa música, apenas mapeia e registra faixas/albuns citados nos vídeos para uma playlist do Spotify.
- Ideal utilizar para canais de divulgação musical.
- Dependente das APIs externas (YouTube, Spotify, Groq/IA).

## Requisitos das variáveis de ambiente

- `CLIENT_ID`/`CLIENT_SECRET`/`REFRESH_TOKEN`: Chaves e token de autenticação do Spotify
- `USER_ID`: ID do usuário no Spotify
- `YT_API_KEY`: API Key do YouTube (Cloud Console)
- Outros parâmetros (Groq, OpenAI, etc) podem ser necessários dependendo do backend IA

## Dependências principais

- [LangChain](https://langchain.com/)
- [Requests](https://docs.python-requests.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [Tqdm](https://github.com/tqdm/tqdm)
- [Pydantic](https://docs.pydantic.dev/)
- [Tenacity (retry)](https://tenacity.readthedocs.io/)

## Licença

[MIT](LICENSE)

---

**Autor:** [bkmoises](https://github.com/bkmoises)


