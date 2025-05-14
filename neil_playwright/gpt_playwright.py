from neil_logger import UniversalLogger
from .gpt_utils import GPTHandler
import imaplib
from datetime import datetime
import time
import re
from .playwright_utils import PlaywrightManager
from pymongo.database import Database


class GPTPlaywright:
    def __init__(self, api_key: str = None, config = None, playwright_manager: PlaywrightManager = None, db: Database = None, logger: UniversalLogger = None):
        self.logger = logger or print
        self.gpt = GPTHandler(api_key, config, logger)
        self.playwright_manager = playwright_manager
        self.db = db
        self.endpoints_db = self.db["endpoints"]
        self.selectors_db = self.db["selectors"]
        self.fieldpolicies_db = self.db["field_policies"]


    # ─────────────────────────────────────────────
    # Selectors Database Helpers
    # ─────────────────────────────────────────────

    # Function to fetch the most recent selectors for a specific category from the database
    def get_selector_doc_from_db(self, category: str) -> dict:
        self.logger.info(f"Fetching selector doc for category: {category}")
        # Fetch the selectors for the given category
        selectors_doc = self.selectors_db.find_one({"category": category})
        if not selectors_doc:
            self.logger.error(f"No selector doc found for category: {category}")
            return {}
        
        return selectors_doc


    # Function to update selectors for a specific category in the database
    def update_selectors_in_db(self, category: str, new_selectors: dict) -> None:
        self.logger.info(f"Updating selectors for category: {category}")
        # Update the existing document for this category or create if it doesn't exist
        self.selectors_db.update_one(
            {"category": category},
            {
                "$set": {
                    "selectors": new_selectors,
                    "timestamp": datetime.now()
                }
            },
            upsert=True
        )
        self.logger.info(f"Successfully updated selectors for category: {category}")


    # ─────────────────────────────────────────────
    # JSON Keys Database Helpers
    # ─────────────────────────────────────────────

    # Function to fetch the most recent JJSON keys for a specific category from the database
    def get_endpoint_doc_from_db(self, endpoint_name: str) -> dict:
        self.logger.info(f"Fetching endpoint doc for: {endpoint_name}")
        # Fetch the selectors for the given category
        endpoint_doc = self.endpoints_db.find_one({"name": endpoint_name})
        if not endpoint_doc:
            self.logger.error(f"No endpoint doc found for: {endpoint_name}")
            return {}
        
        return endpoint_doc


    # Function to update JSON keys for a specific category in the database
    def update_keys_in_db(self, endpoint_name: str, new_keys: dict) -> None:
        self.logger.info(f"Updating keys for endpoint: {endpoint_name}")
        # Update the existing document for this category or create if it doesn't exist
        self.endpoints_db.update_one(
            {"name": endpoint_name},
            {
                "$set": {
                    "keys": new_keys,
                    "timestamp": datetime.now()
                }
            },
            upsert=True
        )
        self.logger.info(f"Successfully updated keys for endpoint: {endpoint_name}")

    # ─────────────────────────────────────────────
    # Prompt Helpers
    # ─────────────────────────────────────────────

    def get_gpt_prompts(self, doc: dict, collection: str) -> tuple[str, str]:
        if collection == "selectors":
            field_docs = self.fieldpolicies_db.find({"selectors": doc.get("category")})
        elif collection == "endpoints":
            field_docs = self.fieldpolicies_db.find({"endpoints": doc.get("name")})
        else:
            field_docs = []
            self.logger.error(f"Invalid collection: {collection}")
            return None, None

        master_prompt = doc.get("master_prompt", None)
        optional_prompts_array = doc.get("optional_prompts", [])

        prompt_pattern = re.compile(r'^(?:\((?P<num>\d+)\))?(?P<field>[^:]+):\s*(?P<desc>.+)$')

        optional_prompts = []

        for prompt in optional_prompts_array:
            match = prompt_pattern.match(prompt)
            if not match:
                continue
            order = int(match.group("num")) if match.group("num") else None
            optional_prompts.append({
                "field_name": match.group("field"),
                "order": order,
                "description": match.group("desc")
            })

        optional_prompts.sort(key=lambda d: (d["order"] is None, d["order"] or 0))
        seen = {p["field_name"] for p in optional_prompts}
        field_prompts_array = list(optional_prompts)

        for field_doc in field_docs:
            field_name = field_doc.get("field_name")
            if field_name not in seen:
                seen.add(field_name)
                field_prompts_array.append({
                    "field_name": field_name,
                    "order": None,
                    "description": field_doc.get("description", "")
                })

        lines = [
            f"{i}. {blk['field_name']} - {blk['description']}"
            for i, blk in enumerate(field_prompts_array, start=1)
        ]

        field_prompts = "\n" + "\n".join(lines)

        total_prompt = master_prompt.replace("{fields}", field_prompts)

        return total_prompt


    # ─────────────────────────────────────────────
    # Selector Helpers
    # ─────────────────────────────────────────────

    # Function to get selectors for a specific category, handling caching and updates
    def get_selectors(self, category: str, html_sample: str = None, force_update: bool = False) -> dict:
        # Get selectors from database
        selector_doc = self.get_selector_doc_from_db(category)
        
        # Check if selectors are out of date (older than a week)
        if selector_doc and selector_doc.get("selectors") and (datetime.now() - selector_doc.get("timestamp", datetime.min)).days < 7 and not force_update:
            self.logger.info(f"Using cached selectors for category: {category}")
            return selector_doc.get("selectors", {})
            
        # If no HTML sample provided, get the default URLs for the category
        if not html_sample:
            try:
                # Get list of URLs to visit
                urls = selector_doc.get("urls", [])
                if isinstance(urls, str):
                    urls = [urls]
                self.logger.info(f"Fetching HTML from {len(urls)} URLs for category {category}")
                
                # Collect HTML from each URL
                html_samples = []
                for url in urls:
                    self.logger.info(f"Loading URL: {url}")
                    
                    # Use load_url with generic body selector
                    self.playwright_manager.load_page("load_url", url, ["settle_html"])
                    
                    html_samples.append(self.playwright_manager.page.content())
                
                # Combine all HTML samples
                html_sample = "\n".join(html_samples)
                    
            except Exception as e:
                self.logger.error(f"Failed to get HTML samples: {e}")
                return {}
            
        try:
            prompt = self.get_gpt_prompts(selector_doc, "selectors")

            # Get new selectors through the router
            new_selectors = self.gpt.request_selectors(html_sample, prompt)
            
            # Update the database with new selectors
            if new_selectors:
                self.update_selectors_in_db(category, new_selectors)
            
            return new_selectors
            
        except Exception as e:
            self.logger.error(f"Failed to generate new selectors: {e}")
            return {}
        

    # ─────────────────────────────────────────────
    # JSON Key Helpers
    # ─────────────────────────────────────────────

    # Function to get keys for a specific category, handling caching and updates
    def get_keys(self, endpoint_name: str, json_sample: str = None, force_update: bool = False) -> dict:
        endpoint_doc = self.get_endpoint_doc_from_db(endpoint_name)

        if endpoint_doc and (datetime.now() - endpoint_doc.get("timestamp", datetime.min)).days < 7 and not force_update:
            self.logger.info(f"Using cached keys for endpoint: {endpoint_name}")
            return endpoint_doc.get("keys", {}), endpoint_doc.get("endpoint_query", None)
        elif not endpoint_doc:
            self.logger.error(f"No endpoint doc found for endpoint: {endpoint_name}")
            return {}, None
        
        endpoint = endpoint_doc.get("endpoint_query", None)
        
        if not json_sample:
            try:
                visit_url = endpoint_doc.get("url", None)
                self.logger.info(f"Fetching JSON keys for endpoint: {endpoint_name}")

                self.logger.info(f"Loading URL: {visit_url}")
                
                # Use load_url with generic body selector
                responses = self.playwright_manager.load_page("load_url", visit_url, ["settle_html"], endpoints = [endpoint])[1]

                if responses:
                    json_sample = responses[endpoint]
                else:
                    self.logger.error(f"No response found for endpoint: {endpoint}")
                    return {}, None
                    
            except Exception as e:
                self.logger.error(f"Failed to get JSON sample: {e}")
                return {}, None
        try:
            prompt = self.get_gpt_prompts(endpoint_doc, "endpoints")

            new_keys = self.gpt.request_keys(json_sample, prompt)

            if new_keys:
                self.update_keys_in_db(endpoint_name, new_keys)

            return new_keys, endpoint
        
        except Exception as e:
            self.logger.error(f"Failed to generate new keys: {e}")
            return {}, None



# ─────────────────────────────────────────────
# Verification detection and handling
# ─────────────────────────────────────────────

class GPTVerification:
    def __init__(self, api_key: str = None, config = None, gmail_email: str = None, gmail_app_password: str = None, logger: UniversalLogger = None):
        self.logger = logger or print
        self.gmail_user = gmail_email
        self.gmail_app_password = gmail_app_password
        self.gpt = GPTHandler(api_key, config, logger)

    # ─────────────────────────────────────────────
    # Verification Helpers
    # ─────────────────────────────────────────────

    # Function to fetch a verification code from Gmail
    def fetch_verification_code(self, sender_domain: str, keyword: str = None, timeout: int = 60):
        M = imaplib.IMAP4_SSL("imap.gmail.com")
        M.login(self.gmail_user, self.gmail_app_password)
        M.select("INBOX")

        deadline = time.time() + timeout
        ids = []

        if not keyword:
            keyword = "verification"

        # 1) poll until we see at least one matching message
        while time.time() < deadline:
            typ, data = M.search(
                None,
                'UNSEEN',
                'OR',
                    'FROM',    f'"{sender_domain}"',
                    'FROM',    f'"voice-noreply@google.com"',
                'OR',
                    'TEXT', '"verification"',
                    'TEXT', f'"{keyword.lower()}"'
            )
            if typ == "OK":
                ids = data[0].split()
                if ids:
                    break
            time.sleep(2)

        if not ids:
            M.logout()
            raise TimeoutError(f"No new OTP from {sender_domain} or Google Voice in {timeout}s")

        latest = ids[-1]  # newest message ID

        # 2) fetch only the plain-text body
        typ, body_data = M.fetch(latest, '(BODY.PEEK[TEXT])')
        M.logout()
        if typ != "OK" or not body_data or not body_data[0]:
            raise RuntimeError("Failed to fetch OTP email body")
        
        body = body_data[0][1].decode(errors="ignore").strip()

        code = self.gpt.request_code(body)

        return code


    
