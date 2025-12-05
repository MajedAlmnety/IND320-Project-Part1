# -*- coding: utf-8 -*-
# Streamlit map view for price areas with MongoDB aggregation and GeoJSON overlays

import os
import json
import requests
import pandas as pd
import streamlit as st
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Map libraries
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, Point
import geopandas as gpd

# Color gradient for choropleth
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Database
from pymongo import MongoClient


# Page configuration and basic CSS styling for the app
st.set_page_config(page_title="Map & Selectors", layout="wide")
load_dotenv()

st.markdown("""
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 0.6rem; }
h1, h2 { margin-bottom: 0.35rem; }
.legend-box {
  background: transparent;
  border: 1px solid #ddd; border-radius: 8px;
  padding: 8px 10px; display: inline-block; font-size: 12px;
}
.legend-bar {
  width: 320px; height: 12px;
  background: linear-gradient(90deg, #440154, #21908C, #FDE725);
  border-radius: 3px; margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# Page title
st.title("Map and Selectors")


# Local GeoJSON file paths for price areas and municipalities
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PRICE_AREAS_FP = os.path.join(DATA_DIR, "price_areas.geojson")     # Price areas NO1..NO5
MUNICIPAL_FP   = os.path.join(DATA_DIR, "municipalities.geojson")  # Municipalities (optional)

# Norway geographic bounds to constrain the map and set minimum zoom level
NORWAY_BOUNDS = [[57.9, 4.0], [71.5, 31.5]]  # [south/west] → [north/east]
MIN_ZOOM = 5  # Do not allow zoom out below this (can increase to 6 if needed)


# Cached helper to load a GeoJSON file from disk
@st.cache_data(show_spinner=False)
def load_geojson_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Cached helper to convert a GeoJSON object to a GeoDataFrame
@st.cache_data(show_spinner=False)
def geojson_to_gdf(geojson_obj: dict) -> gpd.GeoDataFrame:
    feats = geojson_obj.get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {})
        geom = f.get("geometry")
        if geom:
            rows.append({**props, "geometry": shape(geom)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame()

# Detect which property name in the GeoJSON holds the price area code
def detect_price_area_field(props: dict):
    for cand in ("ElSpotOmr", "ELSPOT_OMR", "elspot_omr", "elspotomr", "price_area", "PriceArea", "AREA", "PA"):
        if cand in props:
            return cand
    return None

# Normalize price area strings (uppercase, remove spaces)
def norm_pa(x: str) -> str:
    return str(x).strip().upper().replace(" ", "")


# MongoDB connection (credentials come from .env)
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
except Exception as e:
    st.error(f"Mongo connection failed: {e}")
    st.stop()

DB = os.getenv("MONGO_DB", "elhub_data")
COLL_PROD = os.getenv("COLL_PROD", "production_2022_2024")
COLL_CONS = os.getenv("COLL_CONS", "consumption_2021_2024")


# Load price area polygons from local GeoJSON
try:
    elspot_geojson = load_geojson_file(PRICE_AREAS_FP)
except Exception as e:
    st.error(f"Failed to load price areas file: {e}")
    st.stop()

if not elspot_geojson.get("features"):
    st.error("price_areas.geojson has no features.")
    st.stop()

# Identify correct field name for price area in the GeoJSON properties
PA_FIELD = detect_price_area_field(elspot_geojson["features"][0]["properties"])
if not PA_FIELD:
    st.error(f"Failed to detect price area field in price_areas.geojson. Check property names.")
    st.stop()

# Convert price area GeoJSON to a GeoDataFrame
gdf_elspot = geojson_to_gdf(elspot_geojson)
if gdf_elspot.empty:
    st.error("Failed to convert price_areas.geojson to GeoDataFrame.")
    st.stop()

# Add normalized price area column and list available areas from geometry
gdf_elspot["PA_norm"] = gdf_elspot[PA_FIELD].map(norm_pa)
areas_geo = sorted(gdf_elspot["PA_norm"].unique().tolist())


# Sidebar filters for dataset, price areas, energy groups, and date range
st.sidebar.header("Filters")

dataset = st.sidebar.radio("Dataset", ["Production", "Consumption"], horizontal=True)
collection_name = COLL_PROD if dataset == "Production" else COLL_CONS
coll = client[DB][collection_name]

# Load distinct energy groups and price areas from Mongo to match geometry + data
try:
    groups_all = sorted(coll.distinct("energy_group"))
    areas_db = [norm_pa(x) for x in coll.distinct("price_area")]
    areas_src = sorted(set(areas_db) | set(areas_geo))  # Union between DB and GeoJSON
except Exception:
    groups_all, areas_src = [], areas_geo

areas_sel = st.sidebar.multiselect("Price Areas", areas_src, default=areas_src)
groups_sel = st.sidebar.multiselect("Energy Groups", groups_all, default=groups_all if groups_all else [])

# Date range selection based on available data in Mongo
try:
    min_doc = coll.find_one(sort=[("start_time", 1)])
    max_doc = coll.find_one(sort=[("start_time", -1)])
    min_dt = pd.to_datetime(min_doc["start_time"], utc=True) if min_doc else pd.Timestamp("2022-01-01", tz="UTC")
    max_dt = pd.to_datetime(max_doc["start_time"], utc=True) if max_doc else pd.Timestamp("2024-12-31", tz="UTC")
except Exception:
    min_dt = pd.Timestamp("2022-01-01", tz="UTC")
    max_dt = pd.Timestamp("2024-12-31", tz="UTC")

rng = st.sidebar.date_input(
    "Date range (UTC)",
    value=(min_dt.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)
start_dt = pd.Timestamp(rng[0], tz="UTC").to_pydatetime()
end_dt   = pd.Timestamp(rng[1], tz="UTC").to_pydatetime()

# Optional municipalities overlay (local file, can be heavy)
show_muni = st.sidebar.checkbox("Show municipalities (local, heavy)", value=False,
                                help="May slow down the map. Use only if needed.")

# Session state for user clicks on the map
if "clicks" not in st.session_state:
    st.session_state["clicks"] = []
if st.sidebar.button("Clear clicked points"):
    st.session_state["clicks"].clear()


# MongoDB aggregation: mean value per price area using a pipeline
match_stage = {"start_time": {"$gte": start_dt, "$lt": end_dt}}
if groups_sel:
    match_stage["energy_group"] = {"$in": groups_sel}

# Normalize price_area in Mongo (uppercase + remove spaces) and aggregate by area
pipeline = [
    {"$match": match_stage},
    {"$project": {
        "_id": 0,
        "value": 1,
        "PA_norm": {
            "$replaceAll": {
                "input": {"$toUpper": "$price_area"},
                "find": " ",
                "replacement": ""
            }
        }
    }},
    {"$match": {"PA_norm": {"$in": areas_sel or areas_src}}},
    {"$group": {"_id": "$PA_norm", "mean_value": {"$avg": "$value"}, "count": {"$sum": 1}}},
    {"$project": {"_id": 0, "PA_norm": "$_id", "mean_value": 1, "count": 1}},
    {"$sort": {"PA_norm": 1}},
]

with st.spinner("Aggregating from Mongo…"):
    try:
        agg = list(coll.aggregate(pipeline, allowDiskUse=True))
    except Exception as e:
        st.error(f"Mongo aggregation failed: {e}")
        st.stop()

# Ensure all selected areas are present in the final DataFrame
df_agg = pd.DataFrame(agg)
df_agg = pd.DataFrame({"PA_norm": areas_sel or areas_src}).merge(df_agg, on="PA_norm", how="left")

# Compute color range for choropleth based on mean_value
vmin = float(df_agg["mean_value"].min()) if df_agg["mean_value"].notna().any() else 0.0
vmax = float(df_agg["mean_value"].max()) if df_agg["mean_value"].notna().any() else 1.0
if vmin == vmax:
    vmax = vmin + 1.0

colormap = cm.get_cmap("viridis")

# Map a mean value to a hex color
def color_for_value(val: float) -> str:
    t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    return mcolors.rgb2hex(colormap(t))


# Layout: main map on the left, aggregated table on the right
col_map, col_data = st.columns([8, 3], gap="small")

with col_map:
    st.subheader(f"Elspot Areas — {dataset} mean by area")

    # Lookup table from area code to mean value
    mean_lookup = {r["PA_norm"]: (None if pd.isna(r["mean_value"]) else float(r["mean_value"]))
                   for _, r in df_agg.iterrows()}

    # Styling function for each area polygon
    def style_area(feat):
        area_code = norm_pa(feat.get("properties", {}).get(detect_price_area_field(feat.get("properties", {})) or PA_FIELD))
        val = mean_lookup.get(area_code)
        if val is None:
            return {"fillOpacity": 0.20, "fillColor": "#cccccc", "color": "#666666", "weight": 1.0}
        return {"fillOpacity": 0.65, "fillColor": color_for_value(val), "color": "#444444", "weight": 1.0}

    # Base folium map with bounds and min zoom
    m = folium.Map(
        location=[64.5, 11.0],
        zoom_start=MIN_ZOOM,
        tiles="cartodb positron",
        min_zoom=MIN_ZOOM,   # Prevent zoom out below this level
        max_bounds=True,     # Prevent panning outside bounds
        prefer_canvas=True,  # Faster rendering for many shapes
    )

    # Fit map to Norway bounds only once per session
    if "fit_done" not in st.session_state:
        m.fit_bounds(NORWAY_BOUNDS)
        st.session_state["fit_done"] = True

    # Add price area polygons as GeoJson layer
    folium.GeoJson(
        elspot_geojson,
        name="Elspot mean (by area)",
        style_function=style_area,
        highlight_function=lambda f: {"weight": 2, "color": "#111111"},
        tooltip=folium.features.GeoJsonTooltip(
            fields=[PA_FIELD], aliases=["Price Area"], sticky=False
        ),
    ).add_to(m)

    # Optional municipality overlay
    if show_muni and os.path.exists(MUNICIPAL_FP):
        try:
            muni = load_geojson_file(MUNICIPAL_FP)
            if muni.get("features"):
                muni_fields = muni["features"][0].get("properties", {}).keys()
                label_field = "kommunenavn" if "kommunenavn" in muni_fields else (next(iter(muni_fields)) if muni_fields else None)
                folium.GeoJson(
                    muni,
                    name="Municipalities",
                    style_function=lambda _f: {"fillOpacity": 0.0, "color": "#2c7fb8", "weight": 0.7},
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=[label_field] if label_field else [],
                        aliases=["Municipality"] if label_field else [],
                    ),
                ).add_to(m)
        except Exception as e:
            st.warning(f"Municipality overlay error: {e}")

    # Re-draw previous user clicks as markers
    for lat, lon in st.session_state["clicks"]:
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="#ff6b00",
            fill=True, fill_opacity=0.9,
            popup=f"Clicked: {lat:.5f}, {lon:.5f}",
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    # JavaScript to lock map bounds to Norway region
    m.get_root().html.add_child(folium.Element(f"""
    <script>
      var map = {m.get_name()};
      var bounds = L.latLngBounds(
        [{NORWAY_BOUNDS[0][0]}, {NORWAY_BOUNDS[0][1]}],
        [{NORWAY_BOUNDS[1][0]}, {NORWAY_BOUNDS[1][1]}]
      );
      map.setMaxBounds(bounds);
      map.options.maxBoundsViscosity = 1.0;
    </script>
    """))

    # Show the map and capture the last clicked coordinate
    event = st_folium(
        m,
        height=900,
        width=1500,
        key="map",
        returned_objects=["last_clicked"]
    )

    # Color legend for mean values
    st.markdown(
        f"""
        <div class="legend-box">
          <div style="font-weight:600;">Mean value ({dataset.lower()})</div>
          <div class="legend-bar"></div>
          <div style="display:flex;justify-content:space-between;">
            <span>{vmin:.0f}</span><span>{vmax:.0f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Store last click in session and check which price area it falls into
    if event and isinstance(event, dict) and event.get("last_clicked"):
        lat = event["last_clicked"]["lat"]
        lon = event["last_clicked"]["lng"]
        if (lat, lon) not in st.session_state["clicks"]:
            st.session_state["clicks"].append((lat, lon))
        try:
            pt = Point(lon, lat)
            hit = gdf_elspot[gdf_elspot.geometry.contains(pt)]
            if not hit.empty:
                st.success(f"Clicked point is inside **{norm_pa(hit.iloc[0][PA_FIELD])}**")
            else:
                st.info("Clicked point is outside all price areas.")
        except Exception:
            pass

# Right column: aggregated table and clicked coordinates
with col_data:
    st.subheader("Aggregated table")
    c1, c2 = st.columns(2)
    c1.metric("Selected areas", f"{len(areas_sel or areas_src)}")
    c2.metric("Rows aggregated", f"{int(df_agg['count'].fillna(0).sum()):,}")

    show = df_agg.rename(columns={
        "PA_norm": "Price Area",
        "mean_value": "Mean (value)",
        "count": "Rows"
    })
    st.dataframe(show)

    # Show stored clicked coordinates in a small table
    if st.session_state["clicks"]:
        st.markdown("**Clicked coordinates (session):**")
        st.dataframe(pd.DataFrame(st.session_state["clicks"], columns=["lat", "lon"]))
    else:
        st.caption("No clicks yet — click on the map to add a pin.")
