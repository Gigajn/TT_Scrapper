import time
import random
import asyncio
import sys
import re
try:
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PlaywrightTimeoutError,
        Error as PlaywrightError,
    )
except ImportError:
    sync_playwright = None
    class PlaywrightTimeoutError(Exception): pass
    class PlaywrightError(Exception): pass


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class CaptchaDetectedError(RuntimeError):
    """Raised when TikTok presents a CAPTCHA challenge."""


# ─── Constants ────────────────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# Stealth patches injected before every page load
STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

window.navigator.chrome = { runtime: {} };

Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en']
});

Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5]
});
"""

# Comment text (level-1 comment body)
COMMENT_TEXT_SELECTORS = [
    '[data-e2e="comment-level-1"]',
    '[class*="DivCommentObjectWrapper"] [data-e2e="comment-level-1"]',
    '[class*="DivCommentListContainer"] [data-e2e="comment-level-1"]',
]

# Wrapper that contains both username + text
COMMENT_WRAPPER_SELECTORS = [
    '[class*="DivCommentObjectWrapper"]',
    '[class*="DivCommentListContainer"] [class*="DivCommentObjectWrapper"]',
]

# Panel / container that holds the list
COMMENT_PANEL_SELECTORS = [
    '[class*="DivCommentMain"]',
    '[class*="RightPanelContainer"]',
    '[class*="DivCommentListContainer"]',
]

# Buttons that open the comment panel on some TikTok layouts
COMMENT_TOGGLE_SELECTORS = [
    'button[class*="tux-button__element"]',
    '[class*="tux-button__element"][role="button"]',
    'button:has-text("Comments")',
    'button:has-text("Comment")',
    'button:has-text("Komentar")',
]

# Text prefixes that indicate noise (not real comment content)
_NOISE_PREFIXES = re.compile(
    r"^(lihat |view |see |load more|reply|replies|show \d)",
    re.IGNORECASE,
)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _human_delay(a: float = 0.8, b: float = 2.5) -> None:
    time.sleep(random.uniform(a, b))


def _human_scroll(page) -> None:
    for _ in range(random.randint(2, 5)):
        delta = random.randint(500, 1500)
        page.mouse.wheel(0, delta)
        _human_delay(0.5, 1.5)


def _emit_log(progress_callback, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(message)
    except Exception:
        pass  # Never break scraping because of UI logging failures


def _is_noise(text: str) -> bool:
    """Return True if the text looks like a UI element, not a comment."""
    return bool(_NOISE_PREFIXES.match(text))


# ─── Selector helpers ─────────────────────────────────────────────────────────

def _wait_for_any_selector(page, selectors: list[str], timeout_ms: int) -> bool:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for selector in selectors:
            try:
                if page.locator(selector).first.is_visible(timeout=500):
                    return True
            except (PlaywrightError, PlaywrightTimeoutError):
                continue
        _human_delay(0.1, 0.2)
    return False


def _open_comment_panel_if_needed(page) -> bool:
    """Return True once comments are visible (opening the panel if necessary)."""
    if _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 3000):
        return True

    for selector in COMMENT_TOGGLE_SELECTORS:
        button = page.locator(selector).first
        try:
            if button.is_visible(timeout=1000):
                button.click(timeout=3000)
                _human_delay(0.5, 1.2)

                if _wait_for_any_selector(page, COMMENT_PANEL_SELECTORS, 6000):
                    return True
                if _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 6000):
                    return True
        except (PlaywrightError, PlaywrightTimeoutError):
            continue

    return _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 8000)


# ─── CAPTCHA helpers ──────────────────────────────────────────────────────────

def _is_captcha_page(page) -> bool:
    """Detect CAPTCHA without false-positives when comments are already visible."""
    try:
        # Comments already visible → not blocked
        if _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 800):
            return False

        current_url = (page.url or "").lower()
        if "captcha" in current_url or "/verify" in current_url:
            return True

        detected = page.evaluate(
            """
            () => {
                const hasVisible = (selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden';
                };

                const selectors = [
                    'iframe[src*="captcha"]',
                    'iframe[src*="verify"]',
                    'iframe[src*="challenge"]',
                    'iframe[title*="captcha" i]',
                    '[id*="captcha"]',
                    '[class*="captcha"]',
                    '[data-e2e*="captcha"]',
                    '[class*="verify"]',
                    '[id*="verify"]',
                    '[class*="challenge"]',
                    '[id*="challenge"]'
                ];

                if (selectors.some(hasVisible)) return true;

                const text = (document.body?.innerText || '').toLowerCase();
                const phrases = [
                    'security check',
                    'slide to verify',
                    'drag the slider',
                    'please complete the security check',
                    'verify to continue',
                    'are you human',
                    'complete the puzzle',
                    'verification required'
                ];

                return phrases.some((phrase) => text.includes(phrase));
            }
            """
        )
        return bool(detected)
    except PlaywrightError:
        return False


def _resolve_captcha(page, timeout_ms: int = 180000, progress_callback=None) -> bool:
    """
    Block until the user solves the CAPTCHA and comments become visible,
    or until timeout expires. Returns True on success.
    """
    deadline = time.time() + (timeout_ms / 1000)
    last_announce = 0

    while time.time() < deadline:
        now = time.time()
        if now - last_announce >= 10:
            seconds_left = int(deadline - now)
            _emit_log(
                progress_callback,
                f"Menunggu CAPTCHA diselesaikan... sisa {seconds_left} detik",
            )
            last_announce = now

        if _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 1200):
            return True

        # After user solves the challenge the panel may need reopening.
        _open_comment_panel_if_needed(page)
        _human_delay(0.7, 1.2)

    return False


def _handle_captcha(page, captcha_resolve_count: int, timeout_ms: int, progress_callback) -> int:
    """
    Detect and resolve a CAPTCHA, raising CaptchaDetectedError on failure or
    repeated challenges. Returns updated captcha_resolve_count.
    """
    if not _is_captcha_page(page):
        return captcha_resolve_count

    _emit_log(progress_callback, "CAPTCHA terdeteksi, menunggu penyelesaian manual")

    if not _resolve_captcha(page, timeout_ms=timeout_ms, progress_callback=progress_callback):
        raise CaptchaDetectedError(
            "CAPTCHA detected. Solve it manually in the opened browser, then rerun if needed."
        )

    captcha_resolve_count += 1
    _emit_log(progress_callback, "CAPTCHA selesai, lanjut scraping")

    # Prevent infinite loops when TikTok keeps throwing challenges
    if captcha_resolve_count >= 4:
        raise CaptchaDetectedError(
            "CAPTCHA keeps repeating and comments are still blocked. "
            "Wait a few minutes and try again with a new session/network."
        )

    return captcha_resolve_count


# ─── Scroll helper ────────────────────────────────────────────────────────────

def _scroll_comment_panel(page) -> None:
    """Scroll TikTok's nested comment container; fall back to page scroll."""
    try:
        did_scroll = page.evaluate(
            """
            () => {
                const selectors = [
                    '[class*="DivCommentListContainer"]',
                    '[class*="DivCommentMain"]',
                    '[class*="RightPanelContainer"]'
                ];

                for (const selector of selectors) {
                    const node = document.querySelector(selector);
                    if (!node) continue;

                    const before = node.scrollTop;
                    node.scrollTop = node.scrollHeight;
                    node.dispatchEvent(new Event('scroll', { bubbles: true }));

                    if (node.scrollTop !== before || node.scrollHeight > node.clientHeight) {
                        return true;
                    }
                }

                window.scrollBy(0, 900);
                return false;
            }
            """
        )
        if did_scroll:
            _human_delay(0.4, 1.0)
            return
    except PlaywrightError:
        pass

    _human_scroll(page)


def _pause_video(page) -> None:
    try:
        page.evaluate(
            """
            () => {
                for (const video of document.querySelectorAll('video')) {
                    try { video.pause(); video.muted = true; } catch (_) {}
                }
            }
            """
        )
    except PlaywrightError:
        pass


# ─── Main scraper ─────────────────────────────────────────────────────────────

def scrape_tiktok_comments(url: str, max_comments: int = 50, progress_callback=None) -> list[dict]:
    comments: list[dict] = []
    seen_nodes: set[str] = set()
    captcha_resolve_count = 0

    _emit_log(progress_callback, "Memulai scraper Playwright")

    # Playwright needs ProactorEventLoop for subprocess support on Windows.
    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with sync_playwright() as p:
        _emit_log(progress_callback, "Menjalankan browser")
        browser = p.chromium.launch(
            headless=False,  # headful mode significantly reduces bot detection
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={
                "width": random.randint(1200, 1600),
                "height": random.randint(700, 1000),
            },
            locale="en-US",
        )

        try:
            page = context.new_page()
            _emit_log(progress_callback, "Membuka halaman video TikTok")

            # Inject stealth patches BEFORE any page load
            page.add_init_script(STEALTH_JS)

            _human_delay(1, 3)
            page.goto(url, timeout=60000)
            _emit_log(progress_callback, "Halaman terbuka, mencari panel komentar")

            # ── Phase 1: open comment panel ───────────────────────────────────
            if not _open_comment_panel_if_needed(page):
                _emit_log(progress_callback, "Panel komentar tidak ditemukan")
                return []

            _emit_log(progress_callback, "Panel komentar ditemukan")
            _pause_video(page)
            _emit_log(progress_callback, "Video dijeda saat scraping")

            # ── Phase 2: resolve initial CAPTCHA if present ───────────────────
            captcha_resolve_count = _handle_captcha(
                page, captcha_resolve_count, timeout_ms=180000,
                progress_callback=progress_callback,
            )
            _pause_video(page)

            # ── Phase 3: wait for comments to become visible ──────────────────
            comment_deadline = time.time() + 30
            comments_visible = False

            while time.time() < comment_deadline:
                if _wait_for_any_selector(page, COMMENT_TEXT_SELECTORS, 1200):
                    comments_visible = True
                    break

                captcha_resolve_count = _handle_captcha(
                    page, captcha_resolve_count, timeout_ms=180000,
                    progress_callback=progress_callback,
                )
                _pause_video(page)
                _human_delay(0.4, 0.9)

            if not comments_visible:
                _emit_log(progress_callback, "Timeout saat menunggu komentar muncul")
                return []

            _emit_log(progress_callback, "Komentar mulai terdeteksi, memproses data")
            _human_delay(1, 2)

            # ── Phase 4: scroll & harvest ─────────────────────────────────────
            # Extraction runs entirely inside the browser via page.evaluate() to
            # avoid stale ElementHandle errors caused by TikTok's virtual scroll
            # (React recycles DOM nodes on every scroll, so Playwright handles
            # obtained via query_selector_all() go stale almost immediately).
            retry_idle = 0
            MAX_IDLE = 5

            while len(comments) < max_comments and retry_idle < MAX_IDLE:
                _scroll_comment_panel(page)

                # Pull all currently-rendered comments in one atomic JS round-trip.
                try:
                    raw_items: list[dict] = page.evaluate(
                        """
                        () => {
                            const results = [];
                            const wrappers = document.querySelectorAll(
                                '[class*="DivCommentObjectWrapper"]'
                            );

                            for (const wrapper of wrappers) {
                                // Comment text
                                const commentEl = wrapper.querySelector(
                                    '[data-e2e="comment-level-1"]'
                                );
                                if (!commentEl) continue;

                                const text = (commentEl.innerText || '').trim();
                                if (!text) continue;

                                // Username: try the display-name <p> inside the
                                // username wrapper first, fall back to the <a> href.
                                let username = '';
                                const nameEl = wrapper.querySelector(
                                    '[data-e2e="comment-username-1"] p'
                                );
                                if (nameEl) {
                                    username = (nameEl.innerText || '').trim();
                                }
                                if (!username) {
                                    const linkEl = wrapper.querySelector(
                                        '[data-e2e="comment-username-1"] a'
                                    );
                                    if (linkEl) {
                                        // href is like /@username — strip the leading /@
                                        username = (linkEl.getAttribute('href') || '')
                                                   .replace(/^\/@?/, '') || '';
                                    }
                                }

                                results.push({
                                    account: username || 'unknown',
                                    comment: text,
                                });
                            }

                            return results;
                        }
                        """
                    )
                except PlaywrightError:
                    raw_items = []

                new_this_round = 0
                total_in_dom = len(raw_items)

                for item in raw_items:
                    if len(comments) >= max_comments:
                        break

                    text = (item.get("comment") or "").strip()
                    if not text or _is_noise(text):
                        continue

                    # Dedup by exact content key (account + comment).
                    # DOM-stamp dedup is unreliable when React recycles nodes.
                    dedup_key = f"{item.get('account', '')}|||{text}"
                    if dedup_key in seen_nodes:
                        continue

                    comments.append(
                        {
                            "account": item.get("account") or "unknown",
                            "comment": text,
                        }
                    )
                    seen_nodes.add(dedup_key)
                    new_this_round += 1

                # Idle detection: count rounds where no new comments arrived
                if new_this_round == 0:
                    retry_idle += 1
                else:
                    retry_idle = 0

                _emit_log(
                    progress_callback,
                    f"Progress: {len(comments)}/{max_comments} komentar"
                    f" | dom: {total_in_dom}"
                    f" | baru: {new_this_round}"
                    f" | idle {retry_idle}/{MAX_IDLE}",
                )

                _human_delay(1, 3)

                # Mid-scrape CAPTCHA check — reset idle counter on success
                if _is_captcha_page(page):
                    captcha_resolve_count = _handle_captcha(
                        page, captcha_resolve_count, timeout_ms=120000,
                        progress_callback=progress_callback,
                    )
                    _pause_video(page)
                    retry_idle = 0  # fresh start after solving CAPTCHA

        finally:
            browser.close()
            _emit_log(progress_callback, "Browser ditutup")

    result = comments[:max_comments]
    _emit_log(progress_callback, f"Selesai: {len(result)} komentar berhasil diambil")
    return result