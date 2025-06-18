import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

###  Show my google starred places
# More detailed world map
world = gpd.read_file(r".\ne_10m_admin_0_countries_lakes\ne_10m_admin_0_countries_lakes.shp")
df = pd.read_json('takeout/Maps (your places)/Saved Places.json') # only starred locations, rest are saved in takeout/Saved/...
df = pd.json_normalize(df.features)
# Places without an ID on google maps just give me a URL with coords [0, 0]...
# e.g. http://maps.google.com/?q=44.396403,-93.21813499999999
# q = east,north
# while geometry.coordinates = (north, east)
re_pattern = r"q=(-?\d+\.\d+),(-?\d+\.\d+)"
df[['east', 'north']] = df['properties.google_maps_url'].str.extract(re_pattern).astype(float).fillna(df['geometry.coordinates'].apply(lambda x: pd.Series(x[::-1])))

# Some places don't have coords OR a URL with coordinates...
# e.g. http://maps.google.com/?q=Fotuguan+Park,+Yu+Zhong+Qu,+China&ftid=0x36eccb37b88e6c5f:0x92a89d0704ab5027
# We have to get Google to resolve these types of links..
# Takes a while the first time but caches the result

CACHE_FILE = "coords_cache.json"

def load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def accept_google_consent(driver):
    try:
        agree_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//form//button[contains(., 'I agree') or contains(., 'Ich stimme zu') or contains(., 'Alle akzeptieren')]"))
        )
        agree_button.click()
        print("Clicked consent button")
        time.sleep(5)
    except TimeoutException:
        print("No consent screen detected.")

def get_coords_from_url(driver, url):
    driver.get(url)
    time.sleep(.5)
    accept_google_consent(driver)
    time.sleep(.5)  # Wait for redirect to maps with coordinates
    current_url = driver.current_url
    try:
        path_parts = current_url.split("/@")[1].split(",")
        lat, lon = float(path_parts[0]), float(path_parts[1])
        return lat, lon
    except Exception as e:
        print(f"Could not extract coordinates from: {current_url}")
        return None

def get_coordinates(urls):
    coords = load_cache()

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)

    for url in urls:
        if url in coords:
            print(f"Cached: {url} -> {coords[url]}")
            continue
        print(f"Resolving: {url}")
        result = get_coords_from_url(driver, url)
        if result:
            coords[url] = result
            print(f"Resolved: {url} -> {result}")
            save_cache(coords) # not efficient
        else:
            print(f"Failed to resolve: {url}")

    driver.quit()
    return coords

unknown = df['properties.google_maps_url'][df.east == 0]
coord_dict = get_coordinates(unknown)

df.loc[unknown.index, ['east', 'north']] = [coord_dict[url] for url in unknown]

# Data finally complete ..

# Compute pairwise squared distances (N x N matrix)
points = df[['east', 'north']].to_numpy()
diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # shape: (N, N, 2)
dists = np.linalg.norm(diffs, axis=2)  # Euclidean distances, shape: (N, N)

# Set diagonal to np.inf so a point doesn't consider itself
np.fill_diagonal(dists, np.inf)

# Find the minimum distance and index for each point
min_dists = np.min(dists, axis=1)

def scatter_in_data_units(points, diameters, color='red', dmin=.2, dmax=2, alpha=.8, ax=None):
    if ax is None:
        ax = plt.gca()
    for (xi, yi), r in zip(points, diameters):
        r = max(dmin, r)
        r = min(r, dmax)
        e = Ellipse((yi, xi), width=r, height=r, facecolor='none', edgecolor=color, lw=.7, alpha=alpha)
        ax.add_patch(e)


def scatter_in_data_units_2(points, diameters, color='red', ax=None):
    if ax is None:
        ax = plt.gca()
    for (xi, yi), r in zip(points, diameters):
        lw = max(0.2, r)
        lw = min(lw, 1.2)
        e = Ellipse((yi, xi), width=1.5, height=1.5, facecolor='none', edgecolor=color, lw=lw, alpha=1)
        ax.add_patch(e)

# Light mode
ax = world.plot(color='white', edgecolor='black', alpha=1, lw=.3, figsize=(16,8))
fig = ax.get_figure()
scatter_in_data_units_2(points, min_dists, color='red')
ax.axis('off')
ax.set_xlim(-166.06777070063686, 158.60178343949042)
ax.set_ylim(-58.512240116868426, 78.69029713186694)
fig.canvas.draw()
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('places_light.png', dpi=280, bbox_inches=bbox, pad_inches=0)

# Dark mode
ax = world.plot(color='black', edgecolor='white', alpha=.4, lw=.3, figsize=(16,9))
fig = ax.get_figure()
scatter_in_data_units_2(points, min_dists, color='limegreen')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.axis('off')
ax.set_xlim(-166.06777070063686, 158.60178343949042)
ax.set_ylim(-58.512240116868426, 78.69029713186694)
fig.canvas.draw()
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('places_dark.png', dpi=280, facecolor='black', bbox_inches=bbox, pad_inches=0)