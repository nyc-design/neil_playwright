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
from datetime import datetime, timedelta, timezone
import re
import ast
from html_similarity import style_similarity
from types import SimpleNamespace
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from google.cloud import storage
from pathlib import Path
import uuid
import zipfile
import shutil
import tempfile
from url_normalize import url_normalize



class PlaywrightManager:
    def __init__(self, config = None, logger: UniversalLogger = None):
        self.logger = logger or SimpleNamespace(info = print, warning = print, error = print, debug = print)
        self.data_utils = DataUtils(logger=self.logger)
        self.configuration = self._load_config(config)
        self.profile_path = self.configuration.get("PROFILE_PATH", None)
        self.profile_name = self.configuration.get("PROFILE_NAME", None)
        self.extension_path = self.configuration.get("EXTENSIONS_PATH", None)
        self.is_captcha_extension = self.configuration.get("CAPTCHA_EXTENSION", False)
        self.should_screenshot_errors = self.configuration.get("ERROR_SCREENSHOTS", False)
        self.error_screenshot_path = self.configuration.get("ERROR_SCREENSHOTS_PATH", "screenshots/errors")
        self.video_debug = self.configuration.get("VIDEO_DEBUG", False)
        self.trace_debug = self.configuration.get("TRACE_DEBUG", False)
        self.video_debug_dir = self.configuration.get("VIDEO_DEBUG_PATH", None)
        self.trace_debug_dir = self.configuration.get("TRACE_DEBUG_PATH", None)

        self.temp_dirs = []
        if self.profile_path and self.profile_path.startswith("GCS:"):
            self.gcs_profile = True
            self.gcs_profile_path = self.profile_path
            self.profile_path = self.download_gcs_profile(self.profile_path, self.profile_name)
        else:
            self.gcs_profile = False

        if self.extension_path and self.extension_path.startswith("GCS:"):
            self.extension_path = self.download_extensions(self.extension_path)


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
            screenshot_path = self.screenshot_on_error(final_error)
            self.logger.error(f"Unexpected error during shutdown: {final_error}. Screenshot saved to {screenshot_path}")

        finally:
            if self.gcs_profile:
                self.upload_gcs_profile(self.gcs_profile_path, self.profile_name)
            if self.video_debug or self.trace_debug:
                self.save_debug_files(self.video_debug_dir, self.trace_debug_dir)
            self.data_utils.cleanup_temp()
            if self.temp_dirs:
                for temp_dir in self.temp_dirs:
                    shutil.rmtree(temp_dir)
                self.temp_dirs.clear()

    
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
        profile_dir = self.profile_path if self.persist_session and self.profile_path else os.path.join(tempfile.gettempdir(), "default-chrome-profile")

        self.logger.info(f"[DESKTOP MODE] Launching Chrome with {'persistent session' if self.persist_session else 'ephemeral session'} at: {profile_dir}")

        # Get context args (sets self.abm.geo, user_agent, proxy)
        context_args = self.abm.get_playwright_context(self.playwright.devices)

        args = []

        if incognito:
            args.append("--incognito")
        if self.profile_name:
            args.append(f"--profile-directory={self.profile_name}")
        if self.extension_path:
            args.append(f"--load-extension={self.get_extension_paths()}")

        if self.video_debug:
            additional_args, self.video_temp_path = self.get_video_debug_args()
            context_args.update(additional_args)

        args.append("--no-first-run")
        args.append("--no-default-browser-check")
        args.append("--disable-dev-shm-usage")
        args.append("--disable-features=ProfilePicker")
        args.append("--password-store=basic")
        args.append("--device-scale-factor=1.0000000447034836")
        args.append("--enable-features=WebRTC-H264WithOpenH264FFmpeg")
        args.append("--enable-unsafe-webgpu")
        args.append("--enable-swiftshader")
        args.append("--use-gl=swiftshader")


        context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=headless,
            channel="chrome",
            args=args,
            **context_args  # <- apply proxy, user-agent, locale, geo, etc.
        )

        context = self.abm.add_stealth_scripts(context)

        for p in context.pages():
            p.close()

        page = context.new_page()

        if self.trace_debug:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            
        return context
    

    # Helper function to launch context for mobile mode
    def _launch_mobile_context(self, headless: bool, incognito: bool):
        self.logger.info("[MOBILE MODE] Launching real Chrome in ephemeral mobile context.")

        context_args = self.abm.get_playwright_context(self.playwright.devices)

        args = []

        if incognito:
            args.append("--incognito")

        if self.video_debug:
            additional_args, self.video_temp_path = self.get_video_debug_args()
            context_args.update(additional_args)

        args.append("--no-first-run")
        args.append("--no-default-browser-check")
        args.append("--disable-dev-shm-usage")
        args.append("--disable-features=ProfilePicker")
        args.append("--password-store=basic")
        args.append("--enable-features=WebRTC-H264WithOpenH264FFmpeg")
        args.append("--enable-unsafe-webgpu")
        args.append("--enable-swiftshader")
        args.append("--use-gl=swiftshader")

        context = self.playwright.chromium.launch_persistent_context(
            headless=headless,
            executable_path=self.chrome_path,
            args=args,
            **context_args
        )

        context = self.abm.add_stealth_scripts(context)

        for p in context.pages():
            p.close()

        page = context.new_page()

        if self.trace_debug:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

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
        if self.video_debug or self.trace_debug:
            self.save_debug_files(self.video_debug_dir, self.trace_debug_dir)

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
        e = None

        for attempt in range(retries):
            try:
                with self.expect_responses(endpoints or [], method, timeout) as listeners:
                    self.navigate(current_url, method, locator, hover_and_click = hover_and_click, human_cursor = human_cursor)
                    
                    if not self.test_page_load(tests, before_html, timeout=timeout):
                        self.test_captcha()
                        self.retry_break(attempt)
                        if not self.test_page_load(tests, before_html, timeout=timeout):
                            error = RuntimeError(f"Failed to load page after retry: {self.page.url}")
                            screenshot_path = self.screenshot_on_error(error)
                            self.logger.error(f"Failed to load page after retry: {self.page.url}. Screenshot saved to {screenshot_path}")
                            raise error

                    responses, missing_responses = self.collect_responses(listeners)
                    if missing_responses:
                        error = RuntimeError(f"Failed to collect responses: {missing_responses}")
                        screenshot_path = self.screenshot_on_error(error)
                        self.logger.error(f"Failed to collect responses: {missing_responses}. Screenshot saved to {screenshot_path}")
                        raise error
                    
                    self.responses = responses
                        
                return (self.page, self.responses) if endpoints else (self.page, [])
            
            except ValueError:
                self.logger.error(f"Wrong method passed to load_page: {method}")
                break
            
            except Exception as ex:
                e = ex
                screenshot_path = self.screenshot_on_error(e)
                self.logger.warning(f"Navigation failed for {locator} on attempt {attempt + 1}: {e}. Screenshot saved to {screenshot_path}")
                if not self.persist_session:
                    self.rotate_context()
                self.retry_break(attempt)
                if self.page.url != current_url and attempt < 2:
                    method = "reload"
                else:
                    method = given_method
                    self.page.goto(current_url)
                    self.retry_break(attempt)
                
        if e:
            screenshot_path = self.screenshot_on_error(e)
            self.logger.error(f"All {retries} attempts failed for URL / Selector: {locator}. Screenshot saved to {screenshot_path}")
        else:
            self.logger.error(f"All {retries} attempts failed for URL / Selector: {locator}")

        return self.page, []
    

    # Function to figure out how to navigate
    def navigate(self, current_url: str, method: str, locator: str | Locator, *, hover_and_click: bool = True, human_cursor: bool = True):
        if method == "load_url":
            if isinstance(locator, str) and not locator.startswith(("http://", "https://")):
                locator = url_normalize(locator, default_scheme="https")
            try:
                self.page.goto(locator)
            except Exception as e:
                if isinstance(e, PlaywrightTimeoutError) or "ERR_SSL" in str(e):
                    fallback = url_normalize(locator, default_scheme="http")
                    self.page.goto(fallback)
                else:
                    raise e
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
                error = RuntimeError(f"Failed to open new tab for {locator}")
                screenshot_path = self.screenshot_on_error(error)
                self.logger.error(f"Failed to open new tab for {locator}. Screenshot saved to {screenshot_path}")
                raise error
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
    def test_captcha(self, timeout: float = 200.0):
        if self.abm.is_captcha_present(self.page):
            if self.is_captcha_extension:
                self.precaptcha_url = self.page.url
                start_time = time.time()
                while timeout > (time.time() - start_time):
                    if not self.abm.is_captcha_present(self.page):
                        break
                    time.sleep(1)
                else:
                    if self.precaptcha_url == self.page.url:
                        error = RuntimeError(f"Captcha not solved after {timeout} seconds")
                        screenshot_path = self.screenshot_on_error(error)
                        self.logger.error(f"Captcha not solved after {timeout} seconds. Screenshot saved to {screenshot_path}")
                        raise error
            else:
                error = RuntimeError("Captcha extension is not enabled, but a captcha was detected.")
                screenshot_path = self.screenshot_on_error(error)
                self.logger.error(f"Captcha extension is not enabled, but a captcha was detected. Screenshot saved to {screenshot_path}")
                raise error


    # Function to test the page load
    def test_page_load(self, tests: str | list[str], before_html: str, timeout: float = 30.0):
        if isinstance(tests, str):
            tests = [tests]

        for test in tests:
            try:
                if "text=" in test:
                    text = test.split("text=")[1]
                    pattern = re.compile(re.escape(text), re.IGNORECASE)
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
                screenshot_path = self.screenshot_on_error(e)
                self.logger.error(f"Page load test failed: {test} - {e}. Screenshot saved to {screenshot_path}")
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
        self.center_cursor()


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
            screenshot_path = self.screenshot_on_error(e)
            self.logger.error(f"Failed to return to main tab: {e}. Screenshot saved to {screenshot_path}")

    
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
        end_time = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        diff_pct = 0

        while datetime.now(timezone.utc) < end_time:
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

        error = RuntimeError(f"HTML did not change enough after {timeout} seconds (diff={diff_pct:.2f}%, threshold={threshold}%)")
        screenshot_path = self.screenshot_on_error(error)
        self.logger.error(f"HTML did not change enough after {timeout} seconds. Last diff: {diff_pct}%. Screenshot saved to {screenshot_path}")

        raise error


    # Function to wait for the HTML to settle
    def wait_for_html_settle(self, page: Page, timeout: float = 30.0, threshold: float = 2.0, poll_interval: float = 1.0):
        end_time = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        diff_pct = 100

        while datetime.now(timezone.utc) < end_time:
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


        error = RuntimeError(f"HTML continued to change after {timeout} seconds (diff={diff_pct:.2f}%, threshold={threshold}%)")
        screenshot_path = self.screenshot_on_error(error)
        self.logger.error(f"HTML continued to change after {timeout} seconds. Last diff: {diff_pct}%. Screenshot saved to {screenshot_path}")

        raise error
    
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
            except PlaywrightTimeoutError as e:
                screenshot_path = self.screenshot_on_error(e)
                self.logger.error(f"Timeout waiting for {endpoint}. Screenshot saved to {screenshot_path}")
                responses[endpoint] = None
                missing_responses.append(endpoint)
            except Exception as e:
                screenshot_path = self.screenshot_on_error(e)
                self.logger.error(f"Failed to collect response for {endpoint}: {e}. Screenshot saved to {screenshot_path}")
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
        end_time = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while datetime.now(timezone.utc) < end_time:
            if predicate():
                return
            self.page.wait_for_timeout(poll_interval * 1000)

        error = RuntimeError("Timeout waiting for a predicate")
        screenshot_path = self.screenshot_on_error(error)
        self.logger.error(f"Timeout waiting for a predicate. Screenshot saved to {screenshot_path}")

        raise error
    

    # Function to mimic a micro delay
    def micro_delay(self, min_delay = 0.01, max_delay = 0.05):
        delay = round(random.uniform(min_delay, max_delay), 2)
        time.sleep(delay)


    # Function to mimic random human delay
    def human_delay(self, min_delay = 0.5, max_delay = 1.5):
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


    # Function to center the cursor on the page
    def center_cursor(self):
        vp = self.page.viewport_size or self.page.evaluate("({ width: window.innerWidth, height: window.innerHeight })")
        self.page.mouse.move(vp["width"] / 2, vp["height"] / 2)


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

        error = RuntimeError(f"Failed to scroll to {locator} after {max_scrolls} attempts")
        screenshot_path = self.screenshot_on_error(error)
        self.logger.error(f"Failed to scroll to {locator} after {max_scrolls} attempts. Screenshot saved to {screenshot_path}")

        try:
            locator.scroll_into_view_if_needed()
        except Exception as e:
            screenshot_path = self.screenshot_on_error(e)
            self.logger.error(f"Scroll into view also failed: {e}. Screenshot saved to {screenshot_path}")


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
    # Image Helpers
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
            screenshot_path = self.screenshot_on_error(e)
            self.logger.error(f"Image download error: {e}. Screenshot saved to {screenshot_path}")


    # Function to take a screenshot of webpage
    def take_screenshot(self, filename_prefix: str = None, method: str = None, identifier: str | dict | Locator = None, timeout: float = 10.0, save: bool = True, path: str = None):
        try:
            if not method:
                screenshot_bytes = self.page.screenshot(full_page=False, timeout=timeout*1000)
            elif method == "full_page":
                screenshot_bytes = self.page.screenshot(full_page=True, timeout=timeout*1000)
            elif method == "locator":
                if not isinstance(identifier, Locator):
                    raise ValueError("Locator is required when method is 'locator'")
                screenshot_bytes = identifier.screenshot(timeout=timeout*1000)
            elif method == "selector":
                if not isinstance(identifier, str):
                    raise ValueError("Selector is required when method is 'selector'")
                screenshot_bytes = self.page.locator(identifier).screenshot(timeout=timeout*1000)
            elif method == "clip":
                if not isinstance(identifier, dict):
                    raise ValueError("Bounding box is required when method is 'clip'")
                screenshot_bytes = self.page.screenshot(timeout=timeout*1000, clip=identifier)
            else:
                raise ValueError(f"Invalid method: {method}")
        
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return None

        if save:
            return self.save_screenshot(filename_prefix = filename_prefix, image_bytes = screenshot_bytes, path = path)
        else:
            return screenshot_bytes


    # Function to save a screenshot
    def save_screenshot(self, filename_prefix: str = None, image_bytes: bytes = None, path: str = None) -> str:
        if not filename_prefix:
            raw_url = self.page.url
            parsed = urlparse(raw_url)
            filename_prefix = f"{parsed.netloc}_{parsed.path}"

        filename_prefix = re.sub(r'[^\w\-_.]', '_', filename_prefix)
        
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.png"

        if not image_bytes:
            image_bytes = self.page.screenshot(full_page=True)

        try:
            if not path:
                path = "screenshots"
                Path(path).mkdir(parents=True, exist_ok=True)
                local_path = os.path.join(path, filename)

                with open(local_path, "wb") as f:
                    f.write(image_bytes)
                return local_path
            elif path.startswith("GCS:"):
                if path.endswith("/"):
                    path = path.rstrip("/")
                gcs_path = f"{path}/{filename}"
                blob = self.get_blob(gcs_path)
                blob.upload_from_string(image_bytes, content_type="image/png")
                return f"https://storage.googleapis.com/{gcs_path}"
            else:
                path = path.rstrip("/")
                Path(path).mkdir(parents=True, exist_ok=True)
                local_path = os.path.join(path, filename)

                with open(local_path, "wb") as f:
                    f.write(image_bytes)
                return local_path
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None
            
    
    # Function to call for screenshot on error
    def screenshot_on_error(self, error: Exception):
        if not self.should_screenshot_errors:
            return None

        try:
            error_name = error.__class__.__name__

            msg = str(error) or "error"
            msg_slug = re.sub(r"[^\w\-_.]", "_", msg)[:50]

            page_url = getattr(self.page, "url", "unknown_url")
            url_slug = re.sub(r"[^\w\-_.]", "_", page_url)[:80]

            filename_prefix = f"{error_name}_{msg_slug}_{url_slug}"

            path = self.take_screenshot(method = "full_page", filename_prefix = filename_prefix, path = self.error_screenshot_path)

            return path

        except Exception as e:
            self.logger.warning(f"Failed to capture screenshot on error: {e}")
            return None

    # ─────────────────────────────────────────────
    # Data Extraction Helpers
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
    

    # Function to extract text from html
    def extract_visible_text(self, html: str | list[str], max_length: int = None) -> str:
        if isinstance(html, str):
            html = [html]

        all_text = []

        for item in html:
            try:
                soup = BeautifulSoup(item, "lxml")
            except Exception:
                soup = BeautifulSoup(item, "html.parser")

            for tag in soup(["script", "style", "meta", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            all_text.append(text)

        full_text = "\n\n".join(all_text)

        if max_length:
            full_text = full_text[:max_length]
        
        return full_text
    

    # Function to extract links from html
    def extract_links(self, page: Page = None, base_url: str = None, visible: bool = False):
        if not page:
            page = self.page

        raw_links = page.eval_on_selector_all(
            "a[href]",
            """elements => elements.map(el => {
                const href = el.href || "";
                const text = el.textContent.trim() || "";
                // visible if not display:none and in the layout
                const style = window.getComputedStyle(el);
                const isVisible = style.display !== "none" && el.offsetParent !== null;
                return { href, text, isVisible };
            })"""
        )

        parsed_base = urlparse(base_url).netloc if base_url else None
        filtered = []

        for link in raw_links:
            try:
                href = link["href"]
                text = link["text"]
                is_visible = link["isVisible"]
                
                if not href:
                    continue

                if visible and not is_visible:
                    continue

                if base_url and href.startswith("/"):
                    href = urljoin(base_url, href)

                if parsed_base:
                    netloc = urlparse(href).netloc
                    if netloc and parsed_base not in netloc:
                        continue

                filtered.append({"href": href, "text": text})

            except Exception:
                continue

        return filtered


    # ─────────────────────────────────────────────
    # Cloud Storage Helpers
    # ─────────────────────────────────────────────

    # Function to get a blob from GCS
    def get_blob(self, path: str):
        if not path.startswith("GCS:"):
            raise ValueError("Path must start with GCS:")

        gcs_path = path.split("GCS:")[1]
        gcs_parts = gcs_path.split("/", 1)
        gcs_bucket = gcs_parts[0]
        gcs_path = gcs_parts[1] if len(gcs_parts) > 1 else ""
        
        gcs_credentials_path = self.configuration.get("GCS_CREDENTIALS", None)

        if gcs_credentials_path:
            client = storage.Client.from_service_account_json(gcs_credentials_path)
        else:
            client = storage.Client()

        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)

        return blob
    

    # Function to download extensions from GCS
    def download_extensions(self, gcs_path: str, temp_base: str = None):
        temp_zip_path = self.data_utils.path_to_temp(path=gcs_path, temp_base=temp_base)
        temp_dir = os.path.dirname(temp_zip_path)

        temp_ext_path = os.path.join(temp_dir, "extensions")

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_ext_path)

        return temp_ext_path


    # Function to download a GCS profile
    def download_gcs_profile(self, gcs_path: str, profile_name: str = "Profile 1", temp_base: str = None):
        temp_zip_path = self.data_utils.path_to_temp(path=gcs_path, temp_base=temp_base)
        temp_dir = os.path.dirname(temp_zip_path)

        temp_profile_path = os.path.join(temp_dir, "chrome_profile")

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_profile_path)

        self.original_top_items = set(os.listdir(temp_profile_path))

        self.original_profile_items = set(os.listdir(os.path.join(temp_profile_path, profile_name)))

        return temp_profile_path


    # Function to zip and save a profile to GCS
    def upload_gcs_profile(self, gcs_path: str, profile_name: str = "Profile 1", temp_path: str = None):
        if not temp_path:
            temp_path = self.download_gcs_profile(gcs_path, profile_name)

        self.prune_to_original_items(temp_path, self.original_top_items)
        self.prune_to_original_items(os.path.join(temp_path, profile_name), self.original_profile_items)

        temp_dir = os.path.dirname(temp_path)
        zip_path = self.zip_folder(temp_path, os.path.join(temp_dir, "chrome_profile.zip"))

        blob = self.get_blob(gcs_path)
        blob.upload_from_filename(zip_path)

        return zip_path
        

    # Function to prune to the original items set
    def prune_to_original_items(self, profile_dir: str, original_items: set):
        for item in os.listdir(profile_dir):
            full_path = os.path.join(profile_dir, item)
            if item not in original_items:
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)


    # Function to zip a folder
    def zip_folder(self, folder_path, zip_path):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, folder_path)  # relative inside the zip
                    zipf.write(full_path, arcname)

        return zip_path


    # ─────────────────────────────────────────────
    # Debug Helpers
    # ─────────────────────────────────────────────

    # Function to get debug args
    def get_video_debug_args(self, temp_base: str = None):
        if not self.video_debug:
            return {}
        
        if not temp_base:
            temp_base = tempfile.gettempdir()

        if self.video_debug_dir.startswith("GCS:"):
            temp_dir = tempfile.mkdtemp(dir=temp_base)
            self.temp_dirs.append(temp_dir)
            path = os.path.join(temp_dir, "playwright_videos")
            Path(path).mkdir(parents=True, exist_ok=True)
            
        else:
            path = self.video_debug_dir

        additional_args = {}
        additional_args["record_video_dir"] = path
        additional_args["record_video_size"] = {"width": 1920, "height": 1080}

        return additional_args, path


    # Function to save a trace
    def save_debug_files(self, video_dir: str = None, trace_dir: str = None, temp_base: str = None):
        if not (self.trace_debug or self.video_debug):
            return
        
        if not temp_base:
            temp_base = tempfile.gettempdir()

        if self.trace_debug:
            if trace_dir.startswith("GCS:"):
                temp_dir = tempfile.mkdtemp(dir=temp_base)
                self.temp_dirs.append(temp_dir)
            else:
                temp_dir = trace_dir

            trace_filename = f"playwright_trace_{int(time.time())}.zip"
            trace_temp_path = os.path.join(temp_dir, trace_filename)

            try:
                self.context.tracing.stop(path=trace_temp_path)

                if trace_dir.startswith("GCS:"):
                    if trace_dir.endswith("/"):
                        trace_dir = trace_dir.rstrip("/")
                    trace_gcs_path = f"{trace_dir}/{trace_filename}"
                    self.get_blob(trace_gcs_path).upload_from_filename(trace_temp_path)
                    self.logger.info(f"Trace saved to {trace_gcs_path}")
                else:
                    self.logger.info(f"Trace saved to {trace_temp_path}")

            except Exception as e:
                self.logger.error(f"Failed to save trace: {e}")
                return

        if self.video_debug:
            if video_dir.startswith("GCS:"):
                if video_dir.endswith("/"):
                    video_dir = video_dir.rstrip("/")
                for filename in os.listdir(self.video_temp_path):
                    local_path = os.path.join(self.video_temp_path, filename)
                    if os.path.isdir(local_path):
                        continue
                    try:
                        video_gcs_path = f"{video_dir}/{filename}"
                        self.get_blob(video_gcs_path).upload_from_filename(local_path)
                        self.logger.info(f"Video saved to {video_gcs_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to save video: {e}")
            else:
                self.logger.info(f"Video saved to {self.video_temp_path}")