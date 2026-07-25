# %%
import time
import logging
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.schema import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
class Model():
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.model = self._instance_model()
        
    def _extract_wait_time(self, message: str) -> int:
        """
        Extrai o tempo de espera (em segundos) de uma mensagem de erro textual.
        Caso não seja possível extrair, retorna 120 segundos.
        
        Inputs:
        - message: (str) - Mensagem de erro textual que contém o tempo de espera.
        
        Returns:
        - wait_time: (int) - Tempo de espera em segundos.
        """
        wait_time = 120

        try:
            seconds = str(message).split('Please try again in ')[-1].split('s', 1)[0]
            seconds = float(seconds.replace('.', '').replace('m', '.'))
            wait_time = int(seconds * 60)
        except Exception:
            logging.warning("Falha ao extrair tempo de espera da mensagem de erro.")

        return wait_time
    
    def _instance_model(self) -> callable:
        """
        Cria e retorna um pipeline de extração automática de dados musicais estruturados (artista, faixa, título ou álbum) 
        utilizando um modelo de linguagem natural.

        O prompt orienta o modelo a:
            - Extrair informações sobre músicas ou álbuns a partir de descrições em texto livre.
            - Sempre gerar uma resposta no formato JSON.
            - Preencher campos como "unknown" caso alguma informação não seja identificada.
            - Escolher o formato correto conforme o contexto identificado (música, álbum ou nenhum dos dois).
    
        Returns:
        - callable: Um pipeline composto por prompt, modelo de linguagem e parser de saída JSON.
        """
    
        class MusicDetails(BaseModel):
            artist: str
            track: str
            title: str
            
        llm = ChatGroq(model_name=self.model_name, temperature=self.temperature)
        parser = JsonOutputParser(pydantic_object=MusicDetails)        
        prompt = ChatPromptTemplate.from_messages([            
            ("system", '''You are an assistant whose task is to extract structured data in JSON format from input text.

            - Always respond with a single valid JSON object only, with no explanation.
            - If the text describes a **song**, extract:
                {{
                    "artist": "artist name",
                    "track": "track name",
                    "title": "artist name - track name"
                }}
            - Use "unknown" for any field not explicitly mentioned in the text.
            - If the text is about an **album** (and not about a single song), extract:
                {{
                    "artist": "artist name",
                    "album": "album name"
                }}
            - Use "unknown" for any missing field.
            - Do **not** include fields that were not requested in the chosen format.
            - If neither a song nor album can be identified, return a JSON object with all fields as "unknown".

            **Examples:**

            Input:  
            "The song 'Little Wing' by Jimi Hendrix is amazing."  
            Output:  
            {{
                "artist": "Jimi Hendrix",
                "track": "Little Wing",
                "title": "Jimi Hendrix - Little Wing"
            }}

            Input:  
            "The album 'Hybrid Theory' from Linkin Park defined a generation."  
            Output:  
            {{
                "artist": "Linkin Park",
                "album": "Hybrid Theory"
            }}

            Input:  
            "This is a music channel about heavy metal."  
            Output:  
            {{
                "artist": "unknown",
                "track": "unknown",
                "title": "unknown"
            }}'''), ("user", "{input}")
        ])
        
        return prompt | llm | parser
    
    def ask(self, input: str) -> dict[str, str]:
        """
        Solicita extração dos campos musicais a partir de uma descrição textual usando o modelo LLM.
        Realiza até 3 tentativas automáticas em caso de falhas; após isso, retorna resposta padrão.
        
        Returns:
        - dict[str, str]: Dicionário contendo os campos extraídos ou valores padrão caso não seja possível extrair.
        """
        default_response = {"artist": "Unknown", "track": "Unknown", "title": "Unknown"}

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
        def _attempt() -> dict[str, str]:
            return self.model.invoke(input=input)

        try:
            return _attempt()
        except OutputParserException:
            logging.warning(f"Falha ao converter resposta: {input}")
        except Exception as error:
            wait_time = self._extract_wait_time(str(error))
            logging.error(f"Nova tentativa em {wait_time} segundos.")

            time.sleep(wait_time)
            try:
                return self.model.invoke(input=input)
            except Exception as final_error:
                logging.error(f"Extração de dados falhou para: '{input}'. Erro: {final_error}")

        return default_response
