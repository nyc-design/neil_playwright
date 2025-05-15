from openai import OpenAI
import tiktoken
import json
from bs4 import BeautifulSoup
import re
from neil_logger import UniversalLogger
import os


# ─────────────────────────────────────────────
# GPT Selectors Class
# ─────────────────────────────────────────────

class GPTHandler:
    def __init__(self, api_key: str = None, config = None, logger: UniversalLogger = None):
        self.logger = logger or print
        configuration = self._load_config(config)
        self.upper_model = configuration.get("upper_model", {}).get("name")
        self.lower_model = configuration.get("lower_model", {}).get("name")
        self.upper_token_limit = configuration.get("upper_model", {}).get("limit")
        self.lower_token_limit = configuration.get("lower_model", {}).get("limit")
        self.client = OpenAI(api_key=api_key)


    def _load_config(self, config: str):
        if isinstance(config, str) and os.path.isfile(config):
            with open(config, "r") as f:
                self.logger.info(f"Loading config from {config}")
                configuration = json.load(f)
            return configuration
        elif isinstance(config, dict):
            self.logger.info("Config is already a dict")
            configuration = config
            return configuration
        else:
            raise ValueError("config must be a dict or a valid JSON file path")

# ─────────────────────────────────────────────
# General GPT selector functions
# ─────────────────────────────────────────────

    # Function to make a call to GPT-4 for extracting CSS selectors from HTML
    def request_selectors(self, html_sample: str, prompt: str, model: str = None) -> dict:
        try:
            html_snippet, model = self.preprocess_html(html_sample, self.upper_model)
            system_prompt = """You are an expert web scraper using GPT. Your task is to extract CSS selectors from HTML or Emmet snippets to be use with Playwright's Page.locator().
- Choose selectors that will work time and time again across multiple page reloads and, if applicable, across mutliple pages. Do not over-engineer the selectors. Selectors should not be lengthy, complex, or verbose.
- Return your response as **valid minified JSON**, with all attribute values wrapped in **single quotes** inside double-quoted strings, to prevent escaping issues.
- Do not include comments or extra explanation.
- If you cannot find the elements in the provided HTML, return an empty JSON object {}."""

            response = self.client.responses.create(
                model=model,
                instructions=system_prompt,
                input=prompt + f"\n\nHTML Snippet:\n{html_snippet}",
                text={"format": {"type": "json_object"}},
                temperature=0
            )
            
            content = response.output_text.strip()
            self.logger.info(f"GPT Response: {content}")
            
            return json.loads(content)
                
        except Exception as e:
            self.logger.error(f"GPT call failed: {e}")
            return {}


    def preprocess_html(self, html_sample: str, preferred_model: str = None) -> str:
        if preferred_model == self.upper_model:
            first_limit, second_limit, backup = int(self.upper_token_limit), int(self.lower_token_limit), self.lower_model
        else:
            first_limit, second_limit, backup = int(self.lower_token_limit), int(self.upper_token_limit), self.upper_model

        soup = BeautifulSoup(html_sample, "html.parser")
        body = soup.body or soup

        # 1. Remove truly noisy tags
        for tag in body(["script", "style", "header", "footer", "nav", "aside", "code", "noscript"]):
            tag.decompose()  
        # 2. Handle hidden elements
        for tag in body.find_all(style=True):
            style = tag["style"].lower()
            if "display:none" in style or "visibility:hidden" in style:
                tag.decompose()
        for tag in soup.find_all(attrs={"hidden": True}):
            tag.decompose()

        full_html = self._minify_html(str(body))
        full_tokens = self.count_tokens(full_html)

        if full_tokens < first_limit:
            return full_html, preferred_model
        elif full_tokens < second_limit:
            return full_html, backup

        # 3) Score the sections of the HTML
        sections = body.find_all(["section", "div", "article", "main", "aside"])
        candidates = []

        for section in sections:
            text = section.get_text(separator=" ", strip=True)
            word_count = len(text.split())
            descendant_count = len(section.find_all(["div", "section", "article", "main", "aside", "p", "h1", "h2"]))
            score = (word_count * 1.5) + (descendant_count * 2.0)

            if word_count > 5 or descendant_count > 2:  # small sanity filter
                candidates.append((score, section))

        # Phase 3: Sort and pack
        candidates.sort(key=lambda x: x[0], reverse=True)

        max_limit = max(first_limit, second_limit)

        packed_html = ""
        tokens_used = 0

        for _, section in candidates:
            snippet = str(section)
            snippet_tokens = self.count_tokens(snippet)

            if tokens_used + snippet_tokens > max_limit:
                break

            packed_html += snippet
            tokens_used += snippet_tokens

        packed_html = self._minify_html(packed_html)
        final_model = preferred_model if tokens_used <= first_limit else backup

        return packed_html, final_model
    

    # Function to minify HTML
    def _minify_html(self, html: str) -> str:
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        html = re.sub(r">\s+<", "><", html)
        html = re.sub(r"\s{2,}", " ", html).strip()

        return html
    
# ─────────────────────────────────────────────
# General GPT keys functions
# ─────────────────────────────────────────────

    # Function to make a call to GPT-4 for extracting JSON keys from a JSON object
    def request_keys(self, json_payload: str, prompt: str, model: str = None) -> dict:
        if not model:
            model = self.lower_model

        try:
            system_prompt = """You are an expert web scraper using GPT. You have just received a JSON payload from a web API. Your task is to extract the relevant keys from the JSON object.
- Return your response as **valid minified JSON**, with all attribute values wrapped in **single quotes** inside double-quoted strings, to prevent escaping issues.
- Do not include comments or extra explanation.
- If you cannot find the elements in the provided JSON, return an empty JSON object {}."""

            response = self.client.responses.create(
                model=model,
                instructions=system_prompt,
                input=prompt + f"\n\nJSON Payload:\n{json_payload}",
                text={"format": {"type": "json_object"}},
                temperature=0
            )
            
            content = response.output_text.strip()
            self.logger.info(f"GPT Response: {content}")
            
            return json.loads(content)
                
        except Exception as e:
            self.logger.error(f"GPT call failed: {e}")
            return {}
        
# ─────────────────────────────────────────────
# GPT Verification functions
# ─────────────────────────────────────────────

    def request_code(self, email_body: str, model: str = None) -> str:
        if not model:
            model = self.upper_model
        try:
            prompt = "Extract exactly the 6-digit verification code from this LinkedIn email / text. Return only the code, nothing else:"
            # ask GPT to extract exactly the 6-digit code
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user","content": prompt + f"\n\nEmail Body:\n{email_body}"
                }],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            self.logger.info(f"GPT Response: {content}")
            
            return content
            
        except Exception as e:
            self.logger.error(f"GPT call failed: {e}")
            return {}
        

# ─────────────────────────────────────────────
# Token Counting functions
# ─────────────────────────────────────────────

    def count_tokens(self, text: str, model: str = None) -> int:
        if not model:
            model = self.upper_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
