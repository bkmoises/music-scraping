# %%
import os
import time
import logging
import argparse
import langchain
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# %%
parser = argparse.ArgumentParser(description='Scrape music data from a YouTube channel')
parser.add_argument('--url', type=str, help='URL to scrape chat data from')
parser.add_argument('--full', type=bool, default=False, help='Scrape all videos data if True')
parser.add_argument('--output', type=str, default='musics.xlsx', help='Output file path')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')

# %%
class MusicDetails(BaseModel):
    artist: str
    track: str
    title: str

# %%
class App:
    def __init__(self, url: str, full: bool=False, output: str='musics.xlsx', temperature: float=0.7):
        """
        Initialize the App class.
        :param url: URL of the YouTube page to scrape.
        :param full: Whether to scroll to the bottom of the page to load all videos.
        :param output: Output file name for the results.
        :param temperature: Temperature for the language model.
        """
        if not url.startswith("http"):
            raise ValueError("Invalid URL provided.")
        
        self.url = url
        self.full = full
        self.output = output
        self.output_exists = os.path.exists(output)
        self.dataframe = pd.read_excel(output) if self.output_exists else pd.DataFrame()
        self.channel = url.rsplit("/")[-2].replace("@", "")
        
        self.temperature = temperature
        self.model_name = "llama-3.3-70b-versatile"
        self.llm = ChatGroq(model_name=self.model_name, temperature=self.temperature)
        self.parser = JsonOutputParser(pydantic_object=MusicDetails)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """"You are a JSON extraction assistant. Always respond with a valid JSON using the structure below.\n
            If there are no explicit mentions of a song (artist name and/or track title), return unknown for the fields.

                {{
                    "artist": "artist name here",
                    "track": "track name here",
                    "title": "full title here, artist + track"
                }}"""),
            ("user", "{input}")
        ])
        self.default_response = {
            "artist": "Unknown",
            "track": "Unknown",
            "title": "Unknown",
        }
        self.chain = self.prompt | self.llm | self.parser
        self.titles = self._get_content()
        self.data = []
    
    
    def _parse(self, html: str) -> list[str]:
        """Parse video titles from the page source."""
        logging.info("Parsing HTML content.")
        soup = BeautifulSoup(html, "html.parser")
        titles_elements = soup.find_all(id="video-title")
        
        titles = [title.get_text(strip=True) for title in titles_elements]
        logging.info(f"Extracted {len(titles)} titles from the page.")
        return titles
    

    def _get_content(self) -> List[str]:
        """Scrape the YouTube page and retrieve video titles."""
        logging.info("Initializing WebDriver.")
        options = Options()
        options.add_argument("--headless")

        with webdriver.Firefox(options=options) as driver:
            driver.get(self.url)
            time.sleep(3)

            if self.full:
                logging.info("Scrolling to load all content.")
                last_height = driver.execute_script("return document.documentElement.scrollHeight")

                while True:
                    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                    time.sleep(2)
                    new_height = driver.execute_script("return document.documentElement.scrollHeight")

                    if new_height == last_height:
                        break
                    last_height = new_height

            html = driver.page_source

        return self._parse(html)
    
    
    def _get_wait_time(self, error: str) -> int:
        """Extract wait time from an error message."""
        try:
            seconds = str(error).split('Please try again in ')[-1].split('s', 1)[0]
            seconds = float(seconds.replace('.', '').replace('m', '.'))
            return int(seconds * 60)
        except Exception:
            logging.warning("Failed to extract wait time from error message.")
            return 120

        
    def _ask(self, description: str) -> Dict[str, str]:
        """Send a title description to the language model and parse the response."""
        try:
            return self.chain.invoke({"input": description})
        except OutputParserException:
            logging.warning(f"Failed to parse response for: {description}")
            return self.default_response
        except Exception as error:
            wait_time = self._get_wait_time(error)
            logging.error(f"Error occurred. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
            try:
                return self.chain.invoke({"input": description})
            except Exception:
                logging.error(f"Final failure for input: {description}")
                return self.default_response


    def _save_file(self) -> None:
        """Save the extracted data to an Excel file."""
        if self.output_exists:
            data_df = pd.DataFrame(self.data)
            df = pd.concat([self.dataframe, data_df], ignore_index=True)
        else:
            df = pd.DataFrame(self.data)

        df.to_excel(self.output, index=False)
        logging.info(f"Data successfully saved to {self.output}")
    

    def run(self) -> None:
        """Process all video titles and save the extracted data to an Excel file."""
        logging.info("Starting data extraction.")
        for title in tqdm(self.titles, desc="Extracting", ncols=80):
            if self.channel == 'GreatStonedDragon' and 'dragon' in title.lower():
                title = title.rsplit('||', 1)[0]

                if 'original_title' in self.dataframe.columns and title in self.dataframe['original_title'].values:
                    continue

            response = self._ask(title)

            for key in response.keys():
                response[key] = response[key].title()

            response['original_title'] = title
            response['channel'] = self.channel
            
            self.data.append(response)

        self._save_file()

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    
    app = App(args.url, args.full, args.output, args.temperature)
    app.run()


