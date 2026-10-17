"""Fetch a fresh "Saved Places.json" from Google Takeout.

Usage:
    python fetch_takeout.py              # request a new export, wait, download, extract
    python fetch_takeout.py --zip X.zip  # just extract from an archive you already have

Google has no public Takeout API, so this drives a real Chrome window. It uses
a dedicated profile (.chrome-takeout-profile/) so you only have to sign in the
first time; later runs reuse the session. The only time you should need to
touch the browser is when Google asks for a sign-in or password re-check, and
the script waits for you to finish when that happens.

Chrome is started as a normal process and Selenium attaches to it over the
DevTools port, instead of Selenium launching it. Google refuses sign-ins from a
browser started by chromedriver ("This browser or app may not be secure").
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE / ".chrome-takeout-profile"
DOWNLOAD_DIR = HERE / ".takeout_downloads"
TARGET = HERE / "Takeout" / "Maps (your places)" / "Saved Places.json"

# "local_actions" is Takeout's product id for "Maps (your places)"; the custom/
# URL opens the export page with only that product ticked.
TAKEOUT_URL = "https://takeout.google.com/settings/takeout/custom/local_actions"
MANAGE_URL = "https://takeout.google.com/manage"

# Button labels, English and German (the account UI language decides which).
NEXT_STEP = ["Next step", "Nächster Schritt"]
CREATE_EXPORT = ["Create export", "Export erstellen"]


def find_chrome():
    for name in ("chrome", "google-chrome", "google-chrome-stable", "chromium"):
        path = shutil.which(name)
        if path:
            return path
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    sys.exit("Could not find Chrome. Install it or put chrome on your PATH.")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_chrome():
    port = free_port()
    proc = subprocess.Popen([
        find_chrome(),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ])
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(.5)
    else:
        proc.terminate()
        sys.exit("Chrome did not open its DevTools port.")

    options = Options()
    options.debugger_address = f"127.0.0.1:{port}"
    driver = webdriver.Chrome(options=options)
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    driver.execute_cdp_cmd("Browser.setDownloadBehavior",
                           {"behavior": "allow", "downloadPath": str(DOWNLOAD_DIR)})
    return proc, driver


def host(driver):
    return urlparse(driver.current_url).hostname or ""


def wait_for_takeout(driver, timeout=600):
    """Block while Google shows a sign-in / re-auth page, until we're back on Takeout."""
    if host(driver) == "takeout.google.com":
        return
    print("Google wants you to sign in. Finish it in the Chrome window; I'll wait.")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if host(driver) == "takeout.google.com":
            print("Signed in.")
            return
        time.sleep(1)
    sys.exit("Timed out waiting for sign-in.")


def click_button(driver, labels, timeout=20):
    """Click the first visible button whose text matches one of labels. Returns success."""
    conds = " or ".join(f"normalize-space(.)='{label}'" for label in labels)
    xpath = f"//button[{conds}] | //*[@role='button'][{conds}]"
    deadline = time.time() + timeout
    while time.time() < deadline:
        for el in driver.find_elements(By.XPATH, xpath):
            if el.is_displayed() and el.is_enabled():
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                el.click()
                return True
        time.sleep(.5)
    return False


def ask_user(msg):
    input(f"\n{msg}\nPress Enter here when done... ")


def archive_ids(driver):
    """Export ids listed on the manage page, newest first.

    The manage page only links to a per-export page (manage/archive/<id>);
    the actual download link lives there.
    """
    driver.get(MANAGE_URL)
    wait_for_takeout(driver)
    time.sleep(3)
    ids = []
    for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='manage/archive/']"):
        archive_id = a.get_attribute("href").rstrip("/").rsplit("/", 1)[-1]
        if archive_id not in ids:
            ids.append(archive_id)
    return ids


def archive_download_link(driver, archive_id):
    """The archive's zip download URL, or None if it isn't ready yet."""
    driver.get(f"{MANAGE_URL}/archive/{archive_id}")
    wait_for_takeout(driver)
    time.sleep(3)
    for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='takeout/download']"):
        # "See report" uses the same URL pattern as the zip itself.
        label = (a.get_attribute("aria-label") or "").lower()
        if "report" not in label and "bericht" not in label:
            return a.get_attribute("href")
    return None


def create_export(driver):
    driver.get(TAKEOUT_URL)
    wait_for_takeout(driver)
    time.sleep(3)

    # Guard against the custom URL silently selecting the wrong thing; a full
    # account export would take hours and be many GB.
    checked = driver.execute_script(
        "return [...document.querySelectorAll('input[type=checkbox]:checked')]"
        ".map(c => c.getAttribute('aria-label') || '')")
    if not (len(checked) == 1 and "your places" in checked[0].lower()):
        ask_user("Couldn't confirm the selection. In Chrome, make sure ONLY "
                 "'Maps (your places)' is ticked on the Takeout page.")

    if not click_button(driver, NEXT_STEP):
        ask_user("Couldn't find 'Next step'. Click it in Chrome.")
    # Defaults (send link by email, export once, .zip) are what we want.
    if not click_button(driver, CREATE_EXPORT):
        ask_user("Couldn't find 'Create export'. Click it in Chrome.")
    wait_for_takeout(driver)  # creating an export sometimes asks for re-auth
    print("Export requested.")


def wait_for_link(driver, baseline, timeout_min, poll_s=30):
    """Wait for an archive not in baseline to appear and finish; return its download URL."""
    print(f"Waiting for Google to build the archive (checking every {poll_s}s, "
          f"giving up after {timeout_min} min)...")
    deadline = time.time() + timeout_min * 60
    archive_id = None
    while time.time() < deadline:
        if archive_id is None:
            new = [i for i in archive_ids(driver) if i not in baseline]
            archive_id = new[0] if new else None
        if archive_id is not None:
            link = archive_download_link(driver, archive_id)
            if link:
                return link
        time.sleep(poll_s)
    sys.exit("Timed out. The export may still finish; rerun with --skip-create later, "
             "or download it from the email and use --zip.")


def download(driver, url, timeout=600):
    before = set(DOWNLOAD_DIR.glob("*.zip"))
    driver.get(url)
    print("Downloading (Google may ask for your password again)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = set(DOWNLOAD_DIR.glob("*.zip")) - before
        in_progress = list(DOWNLOAD_DIR.glob("*.crdownload"))
        if new and not in_progress:
            return max(new, key=lambda p: p.stat().st_mtime)
        time.sleep(1)
    sys.exit("Timed out waiting for the download.")


def extract(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith("Saved Places.json")]
        if not members:
            sys.exit(f"No 'Saved Places.json' in {zip_path}.")
        data = zf.read(members[0])

    features = json.loads(data).get("features")
    if features is None:
        sys.exit("Saved Places.json has no 'features'; the format may have changed.")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        shutil.copy2(TARGET, TARGET.with_suffix(".json.bak"))
    TARGET.write_bytes(data)
    print(f"Wrote {TARGET.relative_to(HERE)} ({len(features)} places).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="extract from an existing Takeout zip and exit")
    parser.add_argument("--skip-create", action="store_true",
                        help="don't request a new export; download the newest one on the manage page")
    parser.add_argument("--timeout", type=int, default=60, help="minutes to wait for the export (default 60)")
    args = parser.parse_args()

    if args.zip:
        extract(args.zip)
        return

    proc, driver = launch_chrome()
    try:
        if args.skip_create:
            ids = archive_ids(driver)
            if not ids:
                sys.exit("No exports on the manage page.")
            link = archive_download_link(driver, ids[0])
            if not link:
                sys.exit("The newest export isn't ready (or has expired).")
        else:
            baseline = set(archive_ids(driver))
            create_export(driver)
            link = wait_for_link(driver, baseline, args.timeout)
        extract(download(driver, link))
    finally:
        driver.quit()
        proc.terminate()


if __name__ == "__main__":
    main()
