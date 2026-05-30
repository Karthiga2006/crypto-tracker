
import time
import datetime as dt
import sqlite3
import os
from pathlib import Path

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- Config ----------
HEADLESS = True
POLL_INTERVAL = 60            # seconds between scrapes (adjust as needed)
OUTPUT_DIR = Path("data")
CSV_FILE = OUTPUT_DIR / "crypto_raw.csv"
SQLITE_FILE = OUTPUT_DIR / "crypto_raw.sqlite"
TARGET_URL = "https://coinmarketcap.com/"

# list of top N coins to track (None => track whatever table shows)
TOP_N = 50

# ---------- Helpers ----------
def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def setup_driver(headless=HEADLESS):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    # polite: set a common user agent
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
    # reduce detection
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_window_size(1200, 1000)
    return driver

def parse_price_text(txt):
    """Convert price-like strings to float. Handles commas and $."""
    if txt is None:
        return np.nan
    t = txt.replace("$", "").replace(",", "").replace("—","").strip()
    try:
        return float(t)
    except:
        return np.nan

def parse_percent_text(txt):
    """Convert percent strings like '-2.34%' or '--' to float."""
    if txt is None:
        return np.nan
    t = txt.replace("%", "").replace(",", "").replace("—","").strip()
    try:
        return float(t)
    except:
        return np.nan

# ---------- Scraping ----------
def scrape_cmc_table(driver, top_n=TOP_N):
    """
    Scrape the main table from CoinMarketCap.
    Returns a pandas DataFrame with one snapshot (timestamp).
    """
    driver.get(TARGET_URL)
    wait = WebDriverWait(driver, 15)

    # Wait for table body rows to appear
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except Exception as e:
        # fallback wait a bit more
        time.sleep(3)

    # Use JS to collect visible table rows (more robust than fragile classnames)
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    data = []
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat()

    for i, row in enumerate(rows):
        if top_n and i >= top_n:
            break
        try:
            # The table has many columns; pick the common ones:
            # 1) name + symbol
            # 2) price
            # 3) 24h % 
            # 4) 7d %
            # 5) market cap
            # 6) volume (24h)
            # 7) circulating supply
            # We will use relative xpath inside the row to find columns.
            name_el = row.find_element(By.XPATH, ".//p[@font-weight='bold' or contains(@class,'coin-item-symbol') or contains(@class,'sc-1eb5slv-0')]/..")
            # fallback: capture first link text for name
            try:
                name = name_el.text.splitlines()[0]
            except:
                # alternative attempt
                link = row.find_element(By.XPATH, ".//a[contains(@href,'/currencies/')]")
                name = link.text.strip()
            # symbol
            try:
                symbol = row.find_element(By.XPATH, ".//p[contains(@class,'coin-item-symbol') or contains(@class,'sc-1eb5slv-0')]").text.strip()
            except:
                # fallback: second small text
                symbol = ""
            # price (search for dollar sign)
            price_txt = ""
            try:
                price_el = row.find_element(By.XPATH, ".//td//a[starts-with(normalize-space(.),'$') or contains(.,'$')]")
                price_txt = price_el.text.strip()
            except:
                # alternate column index (CoinMarketCap often puts price in 3rd td)
                tds = row.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 3:
                    price_txt = tds[2].text.strip()
            # 24h and 7d percent - try by % sign
            pct_24h = ""
            pct_7d = ""
            try:
                # collect elements that have % in text
                percent_els = [el for el in row.find_elements(By.XPATH, ".//td//span") if "%" in el.text]
                if len(percent_els) >= 1:
                    pct_24h = percent_els[0].text.strip()
                if len(percent_els) >= 2:
                    pct_7d = percent_els[1].text.strip()
            except:
                pass

            # market cap, volume, supply: try using td positions
            tds = row.find_elements(By.TAG_NAME, "td")
            market_cap = ""
            vol_24h = ""
            supply = ""
            if len(tds) >= 7:
                market_cap = tds[-3].text.strip()
                vol_24h = tds[-2].text.strip()
                supply = tds[-1].text.strip()
            # Clean numeric fields
            price = parse_price_text(price_txt)
            pct24 = parse_percent_text(pct_24h)
            pct7 = parse_percent_text(pct_7d)

            # store
            data.append({
                "ts_utc": now,
                "rank": i+1,
                "name": name,
                "symbol": symbol,
                "price": price,
                "pct_24h": pct24,
                "pct_7d": pct7,
                "market_cap": market_cap,
                "volume_24h": vol_24h,
                "circulating_supply": supply,
            })
        except Exception as e:
            # skip row on parsing error but continue
            print("row parse error:", e)
            continue

    df = pd.DataFrame(data)
    return df

# ---------- Storage ----------
def append_to_csv(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)

def append_to_sqlite(df: pd.DataFrame, sqlite_path: Path, table="prices"):
    conn = sqlite3.connect(sqlite_path)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

# ---------- Analysis helpers ----------
def compute_insights(df: pd.DataFrame):
    """
    Given a DataFrame of raw snapshots (multiple timestamps), return a
    summary DataFrame with analytical columns: returns, moving averages, volatility.
    Assumes df contains columns: ts_utc, symbol, price.
    """
    df2 = df.copy()
    df2['ts'] = pd.to_datetime(df2['ts_utc'])
    # create multiindex: symbol + ts
    df2 = df2.sort_values(['symbol','ts'])
    # compute pct change over previous snapshot per symbol
    df2['pct_change'] = df2.groupby('symbol')['price'].pct_change()
    # compute rolling metrics (example: window=5 snapshots)
    df2['ma_5'] = df2.groupby('symbol')['price'].rolling(window=5, min_periods=1).mean().reset_index(0,drop=True)
    df2['vol_5'] = df2.groupby('symbol')['pct_change'].rolling(window=5, min_periods=1).std().reset_index(0,drop=True)
    return df2

# ---------- Main runner ----------
def run_loop(poll_interval=POLL_INTERVAL, runs=None):
    ensure_dirs()
    driver = setup_driver()
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"[{dt.datetime.utcnow().isoformat()}] Scrape iteration {iteration} ...")
            snapshot = scrape_cmc_table(driver)
            if snapshot.empty:
                print("No rows scraped — check selectors or network.")
            else:
                # Save CSV
                append_to_csv(snapshot, CSV_FILE)
                # Save SQLite
                append_to_sqlite(snapshot, SQLITE_FILE)
                print(f"Saved snapshot with {len(snapshot)} rows.")
                # Example: quick insight printed to console
                # Load last 200 rows (for quick analysis)
                try:
                    raw = pd.read_csv(CSV_FILE)
                    insights = compute_insights(raw)
                    last = insights.groupby('symbol').tail(1)[['symbol','price','pct_change','ma_5','vol_5']]
                    print("Top 5 latest symbols snapshot:")
                    print(last.head(5).to_string(index=False))
                except Exception as e:
                    print("Quick analysis error:", e)

            if runs is not None and iteration >= runs:
                print("Reached run limit; exiting.")
                break
            time.sleep(poll_interval)
    finally:
        driver.quit()

# ---------- Entry point ----------
if __name__ == "__main__":
    # Example: run 10 iterations, every 60s (for testing)
    # In production, remove runs param to run indefinitely
    run_loop(poll_interval=POLL_INTERVAL, runs=None)