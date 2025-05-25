from neil_logger import UniversalLogger
from neil_data_utils import DataUtils
from .gpt_utils import GPTHandler
import imaplib
from datetime import datetime, timezone
import time
import re
import math
from .playwright_utils import PlaywrightManager
from pymongo.database import Database
from collections import defaultdict
from flatten_json import flatten
import json
from types import SimpleNamespace


class GPTPlaywright:
    def __init__(self, api_key: str = None, config = None, playwright_manager: PlaywrightManager = None, db: Database = None, logger: UniversalLogger = None, update_interval: int = 30):
        self.logger = logger or SimpleNamespace(info = print, warning = print, error = print, debug = print)
        self.data_utils = DataUtils(logger = self.logger)
        self.gpt = GPTHandler(api_key, config, logger)
        self.update_interval = update_interval
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
        selectors_doc = self.selectors_db.find_one({"category": category})
        if not selectors_doc:
            self.logger.error(f"No selector doc found for category: {category}")
            return {}
        
        return selectors_doc


    # Function to update selectors for a specific category in the database
    def update_selectors_in_db(self, category: str, new_selectors: dict) -> None:
        self.selectors_db.update_one(
            {"category": category},
            {
                "$set": {
                    "selectors": new_selectors,
                    "timestamp": datetime.now(datetime.UTC)
                }
            },
            upsert=True
        )


    # ─────────────────────────────────────────────
    # JSON Keys Database Helpers
    # ─────────────────────────────────────────────

    # Function to fetch the most recent JJSON keys for a specific category from the database
    def get_endpoint_doc_from_db(self, endpoint_name: str) -> dict:
        endpoint_doc = self.endpoints_db.find_one({"name": endpoint_name})
        if not endpoint_doc:
            self.logger.error(f"No endpoint doc found for: {endpoint_name}")
            return {}
        
        return endpoint_doc


    # Function to update JSON keys for a specific category in the database
    def update_keys_in_db(self, endpoint_name: str, new_keys: dict, sample_json: dict) -> None:
        self.endpoints_db.update_one(
            {"name": endpoint_name},
            {
                "$set": {
                    "keys": new_keys,
                    "sample_json": sample_json,
                    "timestamp": datetime.now(datetime.UTC)
                }
            },
            upsert=True
        )

    # ─────────────────────────────────────────────
    # Prompt Helpers
    # ─────────────────────────────────────────────

    def build_gpt_prompts(self, doc: dict, collection: str) -> tuple[str, str]:
        if collection == "selectors":
            field_docs = list(self.fieldpolicies_db.find({"selectors": doc.get("category")}))
        elif collection == "endpoints":
            field_docs = list(self.fieldpolicies_db.find({"endpoints": doc.get("name")}))
        else:
            self.logger.error(f"Invalid collection: {collection}")
            return None, None

        master_prompt = doc.get("master_prompt") or ""
        optional_prompts_array = doc.get("optional_prompts", [])

        field_doc_map = {f["field_name"]: f for f in field_docs}

        prompt_pattern = re.compile(r'^(?:\((?P<num>\d+)\))?(?P<field>[^:]+):\s*(?P<desc>.+)$')

        items = []
        seen  = set()

        for raw in optional_prompts_array:
            match = prompt_pattern.match(raw)
            if not match:
                continue
            name = match.group("field")
            order = int(match.group("num")) if match.group("num") else None
            description = match.group("desc").strip()

            depends_on = field_doc_map.get(name, {}).get("depends_on") or []

            if isinstance(depends_on, str):
                depends_on = [depends_on]

            items.append({
                "field_name": name,
                "order": order,
                "description": description,
                "depends_on": depends_on
            })

            seen.add(name)

        for field_doc in field_docs:
            field_name = field_doc.get("field_name")
            if field_name in seen:
                continue
            depends_on = field_doc.get("depends_on") or []

            if isinstance(depends_on, str):
                depends_on = [depends_on]

            items.append({
                "field_name": field_name,
                "order": None,
                "description": field_doc.get("description", ""),
                "depends_on": depends_on
            })

        explicit_orders = [item["order"] for item in items if item["order"] is not None]
        max_order = max(explicit_orders) if explicit_orders else 0
        default_priority = max_order + 1

        priority_map = {
            item["field_name"]: (item["order"] or default_priority)
            for item in items
        }

        graph = defaultdict(list)
        in_degree = {item["field_name"]: 0 for item in items}

        names = set(in_degree.keys())
        for item in items:
            for dependency in item.get("depends_on", []):
                if dependency in names:
                    graph[dependency].append(item["field_name"])
                    in_degree[item["field_name"]] += 1
        
        zero_queue = [
            name for name, degree in in_degree.items() if degree == 0
        ]

        zero_queue.sort(key=lambda x: (priority_map[x], x))

        ordered_items = []

        while zero_queue:
            current = zero_queue.pop(0)
            ordered_items.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_queue.append(neighbor)
            
            zero_queue.sort(key=lambda x: (priority_map[x], x))

        if len(ordered_items) < len(items):
            leftover = names - set(ordered_items)
            ordered_items.extend(sorted(leftover, key=lambda x: (priority_map[x], x)))

        field_prompts = []
        name_to_item = {item["field_name"]: item for item in items}
        for idx, name in enumerate(ordered_items, start=1):
            item = name_to_item[name]
            field_prompts.append(f"{idx}. {item['field_name']} - {item['description']}")

        total_prompt = master_prompt.replace("{fields}", "\n".join(field_prompts))

        return total_prompt

    # ─────────────────────────────────────────────
    # Selector Helpers
    # ─────────────────────────────────────────────

    # Function to get selectors for a specific category, handling caching and updates
    def get_selectors(self, category: str, html_sample: str = None, force_update: bool = False) -> dict:
        # Get selectors from database
        selector_doc = self.get_selector_doc_from_db(category)
        
        # Check if selectors are out of date (older than a week)
        if selector_doc and selector_doc.get("selectors") and (datetime.now(datetime.UTC) - selector_doc.get("timestamp", datetime.min.replace(tzinfo=timezone.utc))).days < self.update_interval and not force_update:
            self.logger.info(f"Using cached selectors for category: {category}")
            return selector_doc.get("selectors", {})
        elif not selector_doc:
            self.logger.error(f"No selector doc found for category: {category}")
            return {}
          
        old_selectors = selector_doc.get("selectors", {})
            
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
            prompt = self.build_gpt_prompts(selector_doc, "selectors")

            # Get new selectors through the router
            new_selectors = self.gpt.request_selectors(html_sample, prompt)
            
            # Update the database with new selectors
            if new_selectors:
                if old_selectors:
                    compare_selectors = self.data_utils.compare_jsons(old_selectors, new_selectors, compare_values = True)
                else:
                    compare_selectors = None

                if compare_selectors and compare_selectors["percent_change"] > 10:
                    self.logger.error(f"[{category}] schema shift: +{len(compare_selectors['added'])}/-{len(compare_selectors['removed'])} "
                    f"(pct {compare_selectors['percent_change']:.1f}%) • added={compare_selectors['added']} • removed={compare_selectors['removed']}")

                self.update_selectors_in_db(category, new_selectors)
            
            self.logger.info(f"New selectors generated for category: {category}")
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

        if endpoint_doc and endpoint_doc.get("keys") and (datetime.now(datetime.UTC) - endpoint_doc.get("timestamp", datetime.min.replace(tzinfo=timezone.utc))).days < self.update_interval and not force_update:
            self.logger.info(f"Using cached keys for endpoint: {endpoint_name}")
            return endpoint_doc.get("keys", {}), endpoint_doc.get("endpoint_query", None)
        elif not endpoint_doc:
            self.logger.error(f"No endpoint doc found for endpoint: {endpoint_name}")
            return {}, None
        
        endpoint = endpoint_doc.get("endpoint_query", None)

        old_keys = endpoint_doc.get("keys", {})
        old_json = endpoint_doc.get("sample_json", None)
        
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
            prompt = self.build_gpt_prompts(endpoint_doc, "endpoints")

            flattened_json = self.data_utils.flatten_json(json_sample)
            
            json_payload = json.dumps(flattened_json, separators = (",", ":"))

            new_keys = self.gpt.request_keys(json_payload, prompt)

            if new_keys:
                if old_keys:
                    compare_keys = self.data_utils.compare_jsons(old_keys, new_keys, compare_values = True)
                else:
                    compare_keys = None
                if old_json:
                    compare_jsons = self.data_utils.compare_jsons(old_json, flattened_json)
                else:
                    compare_jsons = None
                
                if ((compare_keys and compare_keys["percent_change"] > 10) or (compare_jsons and compare_jsons["percent_change"] > 10)):
                    self.logger.error(f"[{endpoint_name}] schema shift: +{len(compare_keys['added'])}/-{len(compare_keys['removed'])} "
                    f"(pct {compare_keys['percent_change']:.1f}%) • added={compare_keys['added']} • removed={compare_keys['removed']}")

                self.update_keys_in_db(endpoint_name, new_keys, flattened_json)

            self.logger.info(f"New keys generated for endpoint: {endpoint_name}")
            return new_keys, endpoint
        
        except Exception as e:
            self.logger.error(f"Failed to generate new keys: {e}")
            return {}, None


    #Function to compare JSON response changes
    def compare_json_responses(self, endpoint_name: str, json_response: dict, threshold: float = 50) -> bool:
        endpoint_doc = self.get_endpoint_doc_from_db(endpoint_name)

        if not endpoint_doc:
            self.logger.error(f"No endpoint doc found for endpoint: {endpoint_name}")
            return {}
        
        old_json = endpoint_doc.get("sample_json", {})

        new_json = self.data_utils.flatten_json(json_response)

        comparison = self.data_utils.compare_jsons(old_json, new_json)

        if comparison and comparison["percent_change"] > threshold:
            return True
        else:
            return False

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


    
