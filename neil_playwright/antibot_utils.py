from neil_logger import UniversalLogger
import random
import requests
import json
import os
import glob
import importlib.resources

class AntiBotManager:
    def __init__(self, mode: str = "desktop", config = None, logger: UniversalLogger = None):
        self.logger = logger
        configuration = self._load_config(config)
        self.desktop_device_config = configuration.get("DESKTOP_DEVICE")
        self.mobile_device_config = configuration.get("MOBILE_DEVICE")
        self.proxy_list_config = configuration.get("PROXY_LIST")
        self.captcha_extension_exists = configuration.get("CAPTCHA_EXTENSION")

        self.mode = mode.lower()
        self.proxy_list = self._load_proxies()
        self.devices = self._load_devices()

        if not self.proxy_list:
            self.logger.warning("No proxies loaded! Defaulting to direct connection.")
    

    def _load_config(self, config: str):
        if isinstance(config, str) and os.path.isfile(config):
            with open(config, "r") as f:
                configuration = json.load(f)
            return configuration
        elif isinstance(config, dict):
            configuration = config
            return configuration
        else:
            raise ValueError("config must be a dict or a valid JSON file path")

# ─────────────────────────────────────────────
# Header and context generation
# ─────────────────────────────────────────────

    # Function to load devices
    def _load_devices(self):
        if self.mode == "mobile":
            devices = self._create_list_from_env(self.mobile_device_config)
        else:
            devices = self._create_list_from_env(self.desktop_device_config)
        
        if devices:
            self.logger.info(f"[{self.mode.upper()} MODE] Loaded {len(devices)} devices.")
            return devices
        else:
            self.logger.warning(f"[{self.mode.upper()} MODE] No devices set.")
            self.mode = "desktop"
            return ['Desktop Chrome']


    # Function to load proxy list
    def _load_proxies(self):
        proxies = self._create_list_from_env(self.proxy_list_config)
        if proxies:
            self.logger.info(f"[{self.mode.upper()} MODE] Loaded proxy list with {len(proxies)} proxies.")
            return proxies
        else:
            self.logger.warning(f"[{self.mode.upper()} MODE] No proxies set in PROXY_LIST.")
            return []


    # Function to get geolocation from IP
    def get_geolocation_from_ip(self, proxy: str = None, fallback: dict = None):
        """
        Uses proxy to resolve public IP, then queries ipinfo.io for geolocation.
        """
        try:
            proxies = {"http": proxy, "https": proxy} if proxy else None

            # Step 1: Resolve current public IP through proxy
            ip_resp = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
            ip = ip_resp.json().get("ip")
            if not ip:
                raise RuntimeError("Unable to determine IP through proxy.")

            # Step 2: Lookup geolocation
            geo_resp = requests.get(f"https://ipinfo.io/{ip}/json", timeout=10)
            geo = geo_resp.json()

            if "loc" not in geo:
                raise RuntimeError("Location data not found in response.")

            lat, lon = map(float, geo["loc"].split(","))
            return {
                "latitude": lat,
                "longitude": lon,
                "city": geo.get("city"),
                "region": geo.get("region"),
                "country": geo.get("country"),
                "timezone": geo.get("timezone"),
            }

        except Exception as e:
            self.logger.warning(f"[GeoLookup] Failed to fetch geolocation: {e}")
            return fallback or {
                "latitude": 32.7157,
                "longitude": -117.1611,
                "timezone": "America/Los_Angeles",
            }
    

    # Function to get context for Playwright
    def get_playwright_context(self, playwright_devices: list) -> dict:
        self.device = self.devices[0]
        self.proxy = self.proxy_list[0] if self.proxy_list else None
        self.geo = self.get_geolocation_from_ip(self.proxy)

        device_args = playwright_devices.get(self.device)
        device_args.pop("default_browser_type", None)
        device_args.pop("defaultBrowserType", None)
        device_args.pop("device_scale_factor", None)
        device_args.pop("deviceScaleFactor", None)

        args = {
            **device_args,
            "locale": "en-US",
            "geolocation": {"longitude": self.geo["longitude"], "latitude": self.geo["latitude"], "accuracy": 100},
            "permissions": ["geolocation"],
            "timezone_id": self.geo["timezone"],
        }

        if self.mode != "mobile":
            args["device_scale_factor"] = 1.0000000447034836

        if self.proxy:
            args["proxy"] = {
                "server": 'http://' + self.proxy.split('@')[1],
                "username": self.proxy.split('://')[1].split('@')[0].split(':')[0],
                "password": self.proxy.split('://')[1].split('@')[0].split(':')[1]
            }

        return args

    
    # Function to rotate the identity
    def rotate_identity(self):
        random.shuffle(self.devices)
        random.shuffle(self.proxy_list)


    # Function to add in JS stealth scripts
    def add_stealth_scripts(self, context):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stealth_js_dir = os.path.join(base_dir, "stealth_js")
        scripts = []
        script_names = []

        for filename in sorted(glob.glob(os.path.join(stealth_js_dir, "*.js"))):
            with open(filename, "r", encoding="utf-8") as f:
                scripts.append(f.read())
                script_names.append(os.path.basename(filename))
        
        for script in scripts:
            context.add_init_script(script)

        self.logger.info(f"Added {len(scripts)} stealth scripts to context: {', '.join(script_names)}")
        
        return context
    
# ─────────────────────────────────────────────
# Captcha detection
# ─────────────────────────────────────────────

    # Function to detect captcha
    def is_captcha_present(self, page) -> bool:
        try:
            # preferred: full snapshot API
            html = page.content()
        except Exception as e:
            if "page is navigating" in str(e).lower():
                # fallback to in-browser JS snapshot
                html = page.evaluate("() => document.documentElement.outerHTML")
            else:
                self.logger.warning(f"Failed to get page content: {e}")
                return False

        lower = html.lower()
        
        keywords = [
            "data-sitekey",                     # reCAPTCHA widget
            "/recaptcha/api2/",                 # reCAPTCHA endpoint
            "g-recaptcha",                      # reCAPTCHA init
            "hcaptcha-response",                # hCaptcha widget
            "hcaptcha.js",                      # hCaptcha script
            "funcaptcha.com",                   # FunCaptcha domain
            "/sorry/index",                     # Google “sorry” block page
            "unusual traffic",                  # Google block messaging
            "our systems have detected unusual traffic",
            "verifying you are human",            # Large central text
            "checking your browser before accessing",  # Subtitle/small print
            'id="challenge-form"',               # Form element on interstitial
            'class="challenge-form"',            # Sometimes used instead of id
            "data-translate=\"checking_browser\"",  # Attribute for translation
            "cloudflare turnstile",              # In text sometimes
        ]

        detected = any(token in lower for token in keywords)

        if detected:
            if not self.captcha_extension_exists:
                self.logger.error("CAPTCHA detected on page, but CAPTCHA_EXTENSION does not exist.")
                return False
            else:
                self.logger.warning("CAPTCHA detected on page.")
        else:
            self.logger.debug("No CAPTCHA detected.")

        return detected


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

    # Function to create list from env variable
    def _create_list_from_env(self, env_var: str):
        raw = env_var
        if raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
        else:
            return []

# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

    # Function to test the identity manually
    def test_identity(self, test_url: str = "https://httpbin.org/ip"):
        self.rotate_identity()
        headers = {"User-Agent": self.user_agents[0]}
        proxies = {"http": self.proxy_list[0], "https": self.proxy_list[0]} if self.proxy_list else None

        try:
            # Step 1: Basic proxy test (get external IP)
            ip_resp = requests.get(test_url, headers=headers, proxies=proxies, timeout=10)
            ip = ip_resp.json().get("origin") or ip_resp.json().get("ip")

            if not ip:
                self.logger.warning("Could not determine IP from response.")
                return None

            self.logger.info(f"Current IP: {ip}")

            # Step 2: Get geolocation data using ipinfo.io (no API key needed for limited usage)
            geo_resp = requests.get(f"https://ipinfo.io/{ip}/json", timeout=10)
            geo = geo_resp.json()

            info = {
                "ip": ip,
                "city": geo.get("city"),
                "region": geo.get("region"),
                "country": geo.get("country"),
                "org": geo.get("org"),
                "asn": geo.get("asn", {}).get("asn"),
                "user-agent": headers["User-Agent"]
            }

            self.logger.info(f"Identity Test Result: {info}")
            return info

        except Exception as e:
            self.logger.warning(f"Test identity failed: {e}")
            return None