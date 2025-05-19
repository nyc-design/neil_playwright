from neil_logger import UniversalLogger
from neil_data_utils import DataUtils
import platform
import json
import time
import random
import os
from typing import Optional, Any, Callable
from pathlib import Path
from patchright.sync_api import sync_playwright, Page, expect, Locator, TimeoutError as PlaywrightTimeoutError
from python_ghost_cursor.playwright_sync import create_cursor
from .antibot_utils import AntiBotManager
import requests
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta
import re
import ast
from html_similarity import style_similarity
from types import SimpleNamespace
from urllib.parse import urlparse



class PlaywrightManager:
    def __init__(self, config = None, logger: UniversalLogger = None):
        self.logger = logger or SimpleNamespace(info = print, warning = print, error = print, debug = print)
        self.configuration = self._load_config(config)
        self.chrome_path = self.configuration.get("CHROME_PATH") or self.get_default_chrome_path()
        self.profile_path = self.configuration.get("PROFILE_PATH", None)
        self.profile_name = self.configuration.get("PROFILE_NAME", None)
        self.extension_path = self.configuration.get("EXTENSION_PATH", None)
        self.data_utils = DataUtils(logger=self.logger)


    def launch_playwright(self, mode: str = "desktop", persist_session: bool = False, context_id: str = "default", headless: bool = False, incognito: bool = False):
        self.mode = mode
        self.persist_session = persist_session
        self.context_id = context_id
        self.session_file = f"state_{context_id}.json"
        self.abm = AntiBotManager(mode=self.mode, config=self.configuration, logger=self.logger)
    
        self.playwright = sync_playwright().start()
        self.context = self._init_context(headless, incognito)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.master_tab = self.page
        self.cursor = create_cursor(self.page)
        self.logger.info("Playwright manager initialized successfully")


    def close_playwright(self):
        try:
            if self.context:
                try:
                    self.context.close()
                    self.logger.info("Playwright context closed.")
                except Exception as e:
                    self.logger.warning(f"Context already closed or failed to close cleanly: {e}")
            
            if self.playwright:
                try:
                    self.playwright.stop()
                    self.logger.info("Playwright stopped.")
                except Exception as e:
                    self.logger.warning(f"Failed to stop Playwright cleanly: {e}")
        except Exception as final_error:
            self.logger.error(f"Unexpected error during shutdown: {final_error}")

    
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
    # Context Setup & Rotation
    # ─────────────────────────────────────────────

    # Helper function to launch context for desktop mode
    def _launch_desktop_context(self, headless: bool, incognito: bool):
        profile_dir = self.profile_path if self.persist_session and self.profile_path else "/tmp/default-chrome-profile"

        self.logger.info(f"[DESKTOP MODE] Launching Chrome with {'persistent session' if self.persist_session else 'ephemeral session'} at: {profile_dir}")

        # Get context args (sets self.abm.geo, user_agent, proxy)
        context_args = self.abm.get_playwright_context(self.playwright.devices)

        args=["--no-sandbox","--disable-dev-shm-usage"]
        
        if self.profile_name:
            args.append(f"--profile-directory={self.profile_name}")
        if self.extension_path:
            args.append(f"--load-extension={self.get_extension_paths()}")
        if incognito:
            args.append("--incognito")

        context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=headless,
            executable_path=self.chrome_path,
            args=args,
            **context_args  # <- apply proxy, user-agent, locale, geo, etc.
        )

        return context
    

    # Helper function to launch context for mobile mode
    def _launch_mobile_context(self, headless: bool, incognito: bool):
        self.logger.info("[MOBILE MODE] Launching real Chrome in ephemeral mobile context.")

        context_args = self.abm.get_playwright_context(self.playwright.devices)

        args=["--no-sandbox","--disable-dev-shm-usage"]

        if incognito:
            args.append("--incognito")

        context = self.playwright.chromium.launch_persistent_context(
            headless=headless,
            executable_path=self.chrome_path,
            args=args,
            **context_args
        )

        return context


    # Helper function to initialize the context based on mode
    def _init_context(self, headless: bool, incognito: bool) -> any:
        if self.mode == "desktop":
            return self._launch_desktop_context(headless, incognito)
        else:
            return self._launch_mobile_context(headless, incognito)


    def save_session(self):
        if self.persist_session and self.mode != "desktop":
            self.context.storage_state(path=self.session_file)
    

    # Function to rotate the context
    def rotate_context(self):
        self.abm.rotate_identity()
        self.context.close()

        self.context = self._init_context(headless=False)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.master_tab = self.page


    # ─────────────────────────────────────────────
    # Path Helpers
    # ─────────────────────────────────────────────

    # Function to get the extension paths
    def get_extension_paths(self):
        base_dir = Path(self.extension_path)
        if not base_dir.exists():
            raise ValueError(f"Extension path does not exist: {base_dir}")
        
        # Get subdirectories only
        ext_dirs = [str(d) for d in base_dir.iterdir() if d.is_dir()]
        
        if not ext_dirs:
            raise ValueError(f"No valid extensions found in: {base_dir}")
        
        # Chrome expects comma-separated absolute paths
        return ",".join(ext_dirs)


    # Function to get the default chrome path
    def get_default_chrome_path(self):
        system = platform.system()

        if system == "Windows":
            return r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        elif system == "Darwin":  # macOS
            return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif system == "Linux":
            return "/usr/bin/google-chrome"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
        

    # ─────────────────────────────────────────────
    # URL Checkers
    # ─────────────────────────────────────────────

    # Function to check if we're on a login page
    def is_login_redirect(self) -> bool:
        url = self.page.url.lower()
        title = self.page.title().lower()

        login_indicators = ["login", "signin", "authwall", "authenticate", "log-in", "sign-in"]

        return any(word in url for word in login_indicators) or any(word in title for word in login_indicators)
    

    # Function to check if we're on a verification page
    def is_verification_required(self) -> bool:
        url = self.page.url.lower()
        title = self.page.title().lower()

        login_indicators = ["challenge", "verification", "verify", "verify-email", "verify-phone", "verify-code", "verify-code-sent", "verify-code-sent-email", "verify-code-sent-phone", "checkpoint", "checkpoint-challenge", "checkpoint-challenge-sent", "checkpoint-challenge-sent-email", "checkpoint-challenge-sent-phone"]

        return any(word in url for word in login_indicators) or any(word in title for word in login_indicators)

    # ─────────────────────────────────────────────
    # Navigation Helpers
    # ─────────────────────────────────────────────

    # Function to load a page with retries
    def load_page(self, given_method: str, locator: str | Locator, tests: str | list[str], *, hover_and_click: bool = True, human_cursor: bool = True, endpoints: Optional[list[str]] | str = None, retries: int = 3, timeout: float = 30.0) -> tuple[Page, Optional[dict[str, Optional[dict]]]]:
        current_url = self.page.url  # Store current URL
        before_html = self.page.inner_html("body")
        method = given_method

        for attempt in range(retries):
            try:
                with self.expect_responses(endpoints or [], method, timeout) as listeners:
                    self.navigate(current_url, method, locator, hover_and_click = hover_and_click, human_cursor = human_cursor)
                    
                    if not self.test_page_load(tests, before_html, timeout=timeout):
                        self.test_captcha()
                        self.retry_break(attempt)
                        if not self.test_page_load(tests, before_html, timeout=timeout):
                            raise RuntimeError(f"Failed to load page after retry: {self.page.url}")

                    responses, missing_responses = self.collect_responses(listeners)
                    if missing_responses:
                        raise RuntimeError(f"Failed to collect responses: {missing_responses}")
                    
                    self.responses = responses
                        
                return (self.page, self.responses) if endpoints else (self.page, [])
            
            except ValueError:
                self.logger.error(f"Wrong method passed to load_page: {method}")
                break
            
            except Exception as e:
                self.logger.warning(f"Navigation failed for {locator} on attempt {attempt + 1}: {e}")
                if not self.persist_session:
                    self.rotate_context()
                self.retry_break(attempt)
                if self.page.url != current_url and attempt < 2:
                    method = "reload"
                else:
                    method = given_method
                    self.page.goto(current_url)
                    self.retry_break(attempt)
                

        self.logger.error(f"All {retries} attempts failed for URL / Selector: {locator}")
        return self.page, []
    

    # Function to figure out how to navigate
    def navigate(self, current_url: str, method: str, locator: str | Locator, *, hover_and_click: bool = True, human_cursor: bool = True):
        if method == "load_url":
            self.page.goto(locator)
        elif method == "click":
            if hover_and_click:
                self.scroll_and_hover(locator, human_cursor)
            self.click_locator(locator, human_cursor)
            self.page.wait_for_load_state("load")
            self.page.wait_for_url(lambda url: url != current_url)
        elif method == "click_new_tab":
            if hover_and_click:
                self.scroll_and_hover(locator, human_cursor)
            new_page = self.click_new_tab(locator, human_cursor)
            if not new_page:
                raise RuntimeError(f"Failed to open new tab for {locator}")
            self.page.wait_for_load_state("load")
            self.page.wait_for_url(lambda url: url != current_url)
        elif method == "enter_button":
            if hover_and_click:
                self.scroll_and_hover(locator, human_cursor)
            locator.press("Enter")
            self.page.wait_for_load_state("load")
            self.page.wait_for_url(lambda url: url != current_url)
        elif method == "reload":
            self.page.reload()
            self.page.wait_for_load_state("load")
        else:
            raise ValueError(f"Unknown navigation method: {method}")
        
    
    # Function to poll for Captcha completion
    def test_captcha(self):
        if self.abm.is_captcha_present(self.page):
            self.precaptcha_url = self.page.url
            timeout = 200
            start_time = time.time()
            while timeout > (time.time() - start_time):
                if not self.abm.is_captcha_present(self.page):
                    break
                time.sleep(1)
            else:
                if self.precaptcha_url == self.page.url:
                    raise RuntimeError("Captcha not solved after 200 seconds")


    # Function to test the page load
    def test_page_load(self, tests: str | list[str], before_html: str, timeout: float = 30.0):
        if isinstance(tests, str):
            tests = [tests]

        for test in tests:
            try:
                if "text=" in test:
                    text = test.split("text=")[1]
                    pattern = re.compile(re.escape(text), re.IGNORECASE)
                    print(pattern)
                    self.page.get_by_text(pattern).first.wait_for(timeout=timeout * 1000)
                elif "role=" in test:
                    role = test.split("role=")[1]
                    pattern = re.compile(re.escape(role), re.IGNORECASE)
                    self.page.get_by_role(pattern).first.wait_for(timeout=timeout * 1000)
                elif "placeholder=" in test:
                    placeholder = test.split("placeholder=")[1]
                    pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
                    self.page.get_by_placeholder(pattern).first.wait_for(timeout=timeout * 1000)
                elif "label=" in test:
                    label = test.split("label=")[1]
                    pattern = re.compile(re.escape(label), re.IGNORECASE)
                    self.page.get_by_label(pattern).first.wait_for(timeout=timeout * 1000)
                elif "url=" in test:
                    self.page.wait_for_url(test.split("url=")[1], timeout=timeout * 1000)
                elif "threshold=" in test:
                    self.wait_for_html_change(before_html, threshold=float(test.split("threshold=")[1]), timeout=timeout)
                elif "settle_html" in test:
                    self.wait_for_html_settle(self.page, timeout=timeout)
                elif test == "pass":
                    pass
                else:
                    self.page.locator(test).first.wait_for(timeout=timeout * 1000)
            except Exception as e:
                self.logger.warning(f"Page load test failed: {test} - {e}")
                return False
        
        return True
    

    # Function to help with clicking element and opening new tab
    def click_new_tab(self, locator: str | Locator, human_cursor: bool = True):
        href = locator.get_attribute("href")

        if not href:
            self.logger.warning(f"No href found for selector: {locator}")
            return None
    
        with self.page.expect_popup() as popup_info:
            if self.mode == "mobile":
                self.page.evaluate(f"window.open('{href}', '_blank')")
            else:
                locator.evaluate("el => el.setAttribute('target','_blank')")
                self.click_locator(locator, human_cursor = human_cursor)
        
        new_tab = popup_info.value
        self.logger.info(f"Opened new tab")

        self.reset_page(new_tab)

        return new_tab
    
    # Function to set a tab to the front
    def reset_page(self, page: Page):
        page.bring_to_front()
        page.wait_for_load_state("domcontentloaded")
        page.evaluate("window.focus()")
        self.page = page
        self.cursor = create_cursor(self.page)
        vp = page.viewport_size or page.evaluate("({ width: window.innerWidth, height: window.innerHeight })")
        self.page.mouse.move(vp["width"] / 2, vp["height"] / 2)



    # Function to return to the main tab
    def return_to_main_tab(self, timeout: float = 30.0):
        try:
            current_tab = self.page
            if current_tab != self.master_tab and len(self.context.pages) > 1:
                current_tab.close()

                # Poll until only one tab remains — avoids time.sleep
                for _ in range(20):
                    if len(self.context.pages) == 1:
                        break
                    self.context.pages[0].wait_for_load_state("load", timeout=timeout)

            self.reset_page(self.master_tab)
            self.logger.info("Returned to master tab.")

        except Exception as e:
            self.logger.error(f"Failed to return to main tab: {e}")

    
    # Function to click open a new tab in a with block
    @contextmanager
    def new_tab(self, locator: str | Locator, tests: str | list[str], *, hover_and_click: bool = True, human_cursor: bool = True, endpoints: Optional[list[str]] | str = None, retries: int = 3, timeout: float = 30.0):
        try:
            page, responses = self.load_page("click_new_tab", locator, tests, hover_and_click = hover_and_click, human_cursor = human_cursor, endpoints = endpoints, retries = retries, timeout = timeout)

            yield page, responses

        finally:
            self.return_to_main_tab(timeout=timeout)


    # Function to test how much the HMTL has changed
    def wait_for_html_change(self, before_html: str, threshold: float = 10.0, timeout: float = 10.0, poll_interval: float = 0.5):
        end_time = datetime.now() + timedelta(seconds=timeout)
        diff_pct = 0

        while datetime.now() < end_time:
            try:
                current_html = self.page.inner_html("body")
            except Exception as e:
                self.logger.warning(f"Error reading inner HTML: {e}")
                time.sleep(poll_interval)
                continue

            similarity_score = style_similarity(before_html, current_html)
            diff_pct = (1 - similarity_score) * 100
            if diff_pct >= threshold:
                return diff_pct
                
            time.sleep(poll_interval)

        self.logger.error(f"HTML did not change enough after {timeout} seconds. Last diff: {diff_pct}% against threshold: {threshold}%")
        return diff_pct


    # Function to wait for the HTML to settle
    def wait_for_html_settle(self, page: Page, timeout: float = 30.0, threshold: float = 2.0, poll_interval: float = 1.0):
        end_time = datetime.now() + timedelta(seconds=timeout)
        diff_pct = 100

        while datetime.now() < end_time:
            try:
                first_html = self.page.inner_html("body")
                time.sleep(poll_interval)
                second_html = self.page.inner_html("body")
            except Exception as e:
                self.logger.warning(f"Error reading inner HTML: {e}")
                time.sleep(poll_interval)
                continue

            similarity_score = style_similarity(first_html, second_html)
            diff_pct = (1 - similarity_score) * 100

            if diff_pct <= threshold:
                return diff_pct
                
            time.sleep(poll_interval)

        self.logger.error(f"HTML continued to change after {timeout} seconds. Last diff: {diff_pct}% against threshold: {threshold}%")
        return diff_pct
    
    # ─────────────────────────────────────────────
    # API Interception Helpers
    # ─────────────────────────────────────────────

    @contextmanager
    # Function to begin the API response listeners
    def expect_responses(self, endpoints: list[str] | str, method: str, timeout: float):
        stack = ExitStack()
        listeners = []

        if isinstance(endpoints, str):
            endpoints = [endpoints]

        try:
            for endpoint in endpoints:
                source, search_string = endpoint.split(" {:} ", 1) if " {:} " in endpoint else (None, endpoint)
                header_search = search_string.split(" {>} ")[0] if " {>} " in search_string else search_string
                endpoint_parts = header_search.split(" {+} ") if " {+} " in header_search else [header_search]
                predicate = lambda r, ep=endpoint_parts, src=source: (r.status == 200 and (urlparse(r.url).netloc.find(src) != -1 if src else True) and all(part in r.url for part in ep))
                if method == "click_new_tab":
                    ctx = self.context.expect_event("response", predicate, timeout=timeout * 2 * 1000)
                else:
                    ctx = self.page.expect_response(predicate, timeout=timeout * 2 * 1000)
                listeners.append((endpoint, stack.enter_context(ctx)))

            yield listeners

        finally:
            stack.close()


    # Function to collect the responses from the listeners
    def collect_responses(self, listeners: list) -> dict[str, dict | None]:
        responses = {}
        missing_responses = []
        for endpoint, listener in listeners:
            try:
                response = listener.value
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    responses[endpoint] = response.json()
                elif "text/html" in content_type:
                    html = response.text()
                    json_match = self.extract_json_from_html(html)
                    endpoint_nests = endpoint.split(" {>} ")[1:]
                    for nest in endpoint_nests:
                        json_match = self.data_utils.find_json_by_string(json_match, nest)
                    responses[endpoint] = json_match if json_match else html
                else:
                    responses[endpoint] = response.text()
            except PlaywrightTimeoutError:
                self.logger.error(f"Timeout waiting for {endpoint}")
                responses[endpoint] = None
                missing_responses.append(endpoint)
            except Exception as e:
                self.logger.error(f"Failed to collect response for {endpoint}: {e}")
                responses[endpoint] = None
                missing_responses.append(endpoint)

        if missing_responses:
            self.logger.warning(f"Missing responses for {missing_responses}")
        
        return responses, missing_responses

    # ─────────────────────────────────────────────
    # Human Mimic Helpers
    # ─────────────────────────────────────────────

    # Function to wait for a predicate to be true
    def wait_for(self, predicate: Callable, timeout: float = 10.0, poll_interval: float = 0.5, error_message: str = "Predicate not met after {timeout} seconds"):
        end_time = datetime.now() + timedelta(seconds=timeout)
        while datetime.now() < end_time:
            if predicate():
                return
            self.page.wait_for_timeout(poll_interval * 1000)

        raise RuntimeError(error_message.format(timeout=timeout))
    

    # Function to mimic a micro delay
    def micro_delay(self, min_delay = 0.01, max_delay = 0.05):
        delay = round(random.uniform(min_delay, max_delay), 2)
        time.sleep(delay)


    # Function to mimic random human delay
    def human_delay(self, min_delay = 0.5, max_delay = 2.5):
        delay = round(random.uniform(min_delay, max_delay), 2)
        time.sleep(delay)


    # Function to scroll to locator and hover over it before clicking
    def scroll_and_hover(self, locator: Locator, human_cursor: bool = True):
        self.scroll_to(locator)
        if human_cursor:
            self.cursor_move(locator)
        else:
            locator.hover()
        self.human_delay()


    # Function to click a locator for navigation
    def click_locator(self, locator: Locator, human_cursor: bool = True, modifier: str = None):
        if self.mode == "mobile":
            locator.tap()
        elif human_cursor:
            self.cursor_click(locator, modifier)
        else:
            locator.click(modifiers=modifier)
        self.human_delay()
    

    # Function to move to locator with cursor
    def cursor_move(self, locator: Locator):
        element = locator.element_handle()
        self.cursor.move(element)

    
    # Function to click locator with cursor
    def cursor_click(self, locator: Locator, modifier: str = None):
        element = locator.element_handle()
        if modifier:
            self.page.keyboard.down(modifier)
        self.cursor.click(element)
        if modifier:
            self.page.keyboard.up(modifier)

    
    #Function to mimic page scrolling to element
    def scroll_to(self, locator: Locator, max_scrolls: int = 100, min_scroll: int = 100, max_scroll: int = 500):
        for attempt in range(max_scrolls):
            try:
                expect(locator).to_be_in_viewport(timeout=1)
                self.micro_delay()
                return
            except (PlaywrightTimeoutError, AssertionError):
                box = locator.bounding_box()

                if not box:
                    self.logger.warning(f"Locator {locator} has no bounding box")
                    break

                speed = random.randint(min_scroll, max_scroll)
                if box["y"] <= 0:
                    speed = -speed
                self.micro_delay()
                self.page.mouse.wheel(0, speed)
                self.micro_delay()

        self.logger.error(f"Failed to scroll to {locator} after {max_scrolls} attempts")

        try:
            locator.scroll_into_view_if_needed()
        except Exception as e:
            self.logger.error(f"Scroll into view also failed: {e}")


    # Function to randomly choose an action to mimic behavior
    def random_action(self):
        self.human_delay()
        actions = []

        viewport_size = self.page.viewport_size or {"width": 800, "height": 600}
        width, height = viewport_size["width"], viewport_size["height"]


        # Shared actions
        actions.append(lambda: time.sleep(random.uniform(1.5, 3.5)))

        if self.abm.mode == "mobile":
            actions.extend([
                lambda: self.page.evaluate("""() => {
                    window.scrollBy({ top: Math.floor(Math.random() * 300) + 100, behavior: 'smooth' });
                }""")
            ])
        else:
            # Safe mouse zones — top or bottom center strip
            safe_mouse_regions = [
                (random.randint(width // 3, 2 * width // 3), random.randint(10, 80)),
                (random.randint(width // 3, 2 * width // 3), random.randint(height - 100, height - 20))
            ]
            actions.extend([
                lambda x=x, y=y: self.page.mouse.move(x, y, steps=random.randint(4, 10))
                for x, y in safe_mouse_regions
            ])

            actions.extend([
                lambda: self.page.keyboard.press("ArrowDown"),
                lambda: self.page.keyboard.press("ArrowUp"),
                lambda: self.page.keyboard.press("PageDown"),
                lambda: self.page.keyboard.press("PageUp"),
                lambda: self.page.mouse.move(random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4), steps=random.randint(4, 10)),
                lambda: self.page.mouse.wheel(0, random.randint(-500, 500)),
                lambda: self.page.evaluate("() => window.dispatchEvent(new Event('blur'))"),
                lambda: self.page.evaluate("() => window.dispatchEvent(new Event('focus'))")
            ])
        
        random.choice(actions)()
        self.human_delay()


    # Function to create a random human break
    def random_break(self, counter: int, min_items=30, max_items=60):
        threshold = random.randint(min_items, max_items)
        if counter % threshold == 0 and counter != 0:
            duration = random.randint(30, 180)
            self.logger.info(f"Taking break for {duration} seconds after {counter} items...")
            time.sleep(duration)


    # Function to mimic a small break before retrying link
    def retry_break(self, attempt: int, base: int = 5, max_delay: int = 60):
        delay = min(base * (2 ** attempt), max_delay)
        time.sleep(delay)

    # ─────────────────────────────────────────────
    # Image Download
    # ─────────────────────────────────────────────

    # Function to download an image from a selector
    def download_image(self, selector: str, path: str):
        try:
            url = self.page.get_attribute(selector, "src")
            if not url:
                self.logger.warning(f"No image src found for: {selector}")
                return
        
            headers = {"User-Agent": self.abm.user_agents[0]}
            proxy = self.abm.proxy_list[0] if self.abm.proxy_list else None
            proxies = {"http": proxy, "https": proxy} if proxy else None

            r = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                self.logger.info(f"Image saved: {path}")
            else:
                self.logger.warning(f"Image request failed: {r.status_code}")
        except Exception as e:
            self.logger.error(f"Image download error: {e}")


    # Function to take a screenshot of webpage
    def take_screenshot(self, filename: str = "screenshot.png", full_page: bool = True):
        try:
            self.page.screenshot(path=filename, full_page=full_page)
            self.logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")

    # ─────────────────────────────────────────────
    # JSON Extraction Helpers
    # ─────────────────────────────────────────────

    # Function to extract JSON data from a response
    def extract_json_from_html(self, html: str) -> list:
        results = []

        for match in re.finditer(r'<script[^>]+type="application/json"[^>]*>(.*?)</script>', html, re.S):
            text = match.group(1).strip()
            try:
                results.append(json.loads(text))
            except json.JSONDecodeError:
                pass
        
        match = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        if match:
            text = match.group(1).strip()
            try:
                results.append(json.loads(text))
            except json.JSONDecodeError:
                pass
        
        for tag in re.finditer(r'<script[^>]*>(.*?)</script>', html, re.S):
            for match in re.finditer(r'\{[^{}]*\}', tag.group(1)):
                blob = match.group(0)
                if ":" not in blob:
                    continue
                try:
                    results.append(json.loads(blob))
                except json.JSONDecodeError:
                    try:
                        results.append(ast.literal_eval(blob))
                    except Exception:
                        pass
        
        return results
    


