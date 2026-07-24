import logging
import sys


def setup_logging():
    """
    Configura o sistema de logging com níveis apropriados.
    
    Define:
    - Nível INFO para o logger raiz
    - Formato: timestamp - nível - mensagem
    - Nível WARNING para 'langchain'
    - Nível WARNING para 'httpx'
    
    Compatível com Jupyter Notebook.
    """
    # Remove handlers existentes para evitar conflitos
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Cria novo handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    # Configura logger raiz
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    # Configura loggers específicos
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Criar um logger padrão para uso fácil
logger = logging.getLogger(__name__)
