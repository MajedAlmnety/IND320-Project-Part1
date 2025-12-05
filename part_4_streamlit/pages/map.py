import os
import json
import requests
import pandas as pd
import streamlit as st
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Map stack
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, Point
import geopandas as gpd

# Colors / legend
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Mongo
from pymongo import MongoClient


# -----------------------------
# PAGE & ENV
# -----------------------------
st.set_page_config(page_title="Map & Selectors", layout="wide")
load_dotenv()

# 🔧 CSS لتقليل الفراغات ووضع الـLegend تحت الخريطة
st.markdown("""
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 0.6rem; }
h1, h2 { margin-bottom: 0.35rem; }
.legend-box {
  background: rgba(255,255,255,0.95);
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

st.title("🗺️ Map and Selectors")


# -----------------------------
# CONSTANTS (GEO SOURCES)
# -----------------------------
ELSPOT_URL = (
    "https://nve.geodataonline.no/arcgis/rest/services/Mapservices/Elspot/MapServer/0/"
    "query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
)
# Optional municipalities overlay (محلي أو عبر URL احتياطي)

KOMMUNE_LOCAL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "municipalities.geojson")

# حدود النرويج (تقريبية) لحصر الخريطة والسماح بالتكبير فقط
NORWAY_BOUNDS = [[57.9, 4.0], [71.5, 31.5]]  # [S/W] → [N/E]
MIN_ZOOM = 5  # لا يُسمح بالتصغير أكثر من هذا


# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def geojson_to_gdf(geojson_obj):
    feats = geojson_obj.get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {})
        geom = f.get("geometry")
        if geom:
            rows.append({**props, "geometry": shape(geom)})
    if not rows:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

def detect_price_area_field(props: dict):
    for cand in ("ElSpotOmr", "ELSPOT_OMR", "elspot_omr", "elspotomr", "price_area", "PriceArea"):
        if cand in props:
            return cand
    return None

def norm_pa(x: str) -> str:
    return str(x).strip().upper().replace(" ", "")


# -----------------------------
# MONGO CONNECTION
# -----------------------------
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
except Exception as e:
    st.error(f"❌ Mongo connection failed: {e}")
    st.stop()

DB = os.getenv("MONGO_DB", "elhub_data")
COLL_PROD = os.getenv("COLL_PROD", "production_2022_2024")
COLL_CONS = os.getenv("COLL_CONS", "consumption_2021_2024")


# -----------------------------
# LOAD GEO
# -----------------------------
with st.spinner("Loading NVE Elspot areas…"):
    elspot_geojson = load_geojson(ELSPOT_URL)

feats = elspot_geojson.get("features", [])
if not feats:
    st.error("Could not load Elspot areas.")
    st.stop()

props0 = feats[0].get("properties", {})
PA_FIELD = detect_price_area_field(props0)
if not PA_FIELD:
    st.error(f"Could not detect the price-area field. Available: {tuple(props0.keys())}")
    st.stop()

gdf_elspot = geojson_to_gdf(elspot_geojson)
if gdf_elspot.empty:
    st.error("Failed to parse GeoJSON to GeoDataFrame.")
    st.stop()

gdf_elspot["PA_norm"] = gdf_elspot[PA_FIELD].map(norm_pa)
areas_geo = sorted(gdf_elspot["PA_norm"].unique().tolist())


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

dataset = st.sidebar.radio("Dataset", ["Production", "Consumption"], horizontal=True)
collection_name = COLL_PROD if dataset == "Production" else COLL_CONS
coll = client[DB][collection_name]

try:
    groups_all = sorted(coll.distinct("energy_group"))
    areas_db = [norm_pa(x) for x in coll.distinct("price_area")]
    areas_src = sorted(set(areas_db) | set(areas_geo))
except Exception:
    groups_all, areas_src = [], areas_geo

areas_sel = st.sidebar.multiselect("Price Areas", areas_src, default=areas_src)
groups_sel = st.sidebar.multiselect("Energy Groups", groups_all, default=groups_all if groups_all else [])

# Dates
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
end_dt = pd.Timestamp(rng[1], tz="UTC").to_pydatetime()

show_muni = st.sidebar.checkbox("Show municipalities (bonus)", value=False)

# Clicks
if "clicks" not in st.session_state:
    st.session_state["clicks"] = []
if st.sidebar.button("Clear clicked points"):
    st.session_state["clicks"].clear()


# -----------------------------
# QUERY MONGO (mean per area)
# -----------------------------
match = {"start_time": {"$gte": start_dt, "$lt": end_dt}}
if groups_sel:
    match["energy_group"] = {"$in": groups_sel}

proj = {"_id": 0, "price_area": 1, "value": 1}
cur = coll.find(match, proj, batch_size=5000)
df_raw = pd.DataFrame(list(cur))
if not df_raw.empty:
    df_raw["PA_norm"] = df_raw["price_area"].map(norm_pa)
else:
    df_raw = pd.DataFrame(columns=["PA_norm", "value"])

if areas_sel:
    df_raw = df_raw[df_raw["PA_norm"].isin(areas_sel)]

df_agg = (
    df_raw.groupby("PA_norm", as_index=False)
          .agg(mean_value=("value", "mean"), count=("value", "size"))
)
df_agg = pd.DataFrame({"PA_norm": areas_sel or areas_src}).merge(df_agg, on="PA_norm", how="left")

# Color scale range
vmin = float(df_agg["mean_value"].min()) if df_agg["mean_value"].notna().any() else 0.0
vmax = float(df_agg["mean_value"].max()) if df_agg["mean_value"].notna().any() else 1.0
if vmin == vmax:
    vmax = vmin + 1.0

# Viridis colormap
colormap = cm.get_cmap("viridis")
def color_for_value(val: float) -> str:
    t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    return mcolors.rgb2hex(colormap(t))


# -----------------------------
# MAP + CHOROPLETH (مقيّدة بالنرويج)
# -----------------------------
# عمود أعرض للخريطة بحيث تحتل أغلب الصفحة، والجدول في عمود جانبي ضيّق
col_map, col_data = st.columns([7, 3], gap="small")

with col_map:
    st.subheader(f"Elspot Areas — {dataset} mean by area")

    mean_lookup = {r["PA_norm"]: (None if pd.isna(r["mean_value"]) else float(r["mean_value"]))
                   for _, r in df_agg.iterrows()}

    def style_area(feat):
        area_code = norm_pa(feat.get("properties", {}).get(PA_FIELD))
        val = mean_lookup.get(area_code)
        if val is None:
            return {"fillOpacity": 0.20, "fillColor": "#cccccc", "color": "#666666", "weight": 1.0}
        return {"fillOpacity": 0.65, "fillColor": color_for_value(val), "color": "#444444", "weight": 1.0}

    # ⬅️ خريطة مقيّدة بالنرويج: min_zoom + max_bounds + fit_bounds
    m = folium.Map(
        location=[64.5, 11.0],
        zoom_start=MIN_ZOOM,
        tiles="cartodb positron",
        min_zoom=MIN_ZOOM,   # لا يسمح بالتصغير أقل من هذا
        max_bounds=True      # يمنع الخروج خارج الحدود
    )
    m.fit_bounds(NORWAY_BOUNDS)  # إطّار النرويج بالكامل

    # طبقة الألوان حسب المتوسط
    folium.GeoJson(
        elspot_geojson,
        name="Elspot mean (by area)",
        style_function=style_area,
        highlight_function=lambda f: {"weight": 2, "color": "#111111"},
        tooltip=folium.features.GeoJsonTooltip(
            fields=[PA_FIELD], aliases=["Price Area"], sticky=False
        ),
    ).add_to(m)

    # (اختياري) طبقة البلديات
    if show_muni:
        muni = None
        if os.path.exists(KOMMUNE_LOCAL):
            try:
                with open(KOMMUNE_LOCAL, "r", encoding="utf-8") as f:
                    muni = json.load(f)
            except Exception as e:
                st.warning(f"Municipality local file error: {e}")
        if muni is None:
            try:
                muni = load_geojson(KOMMUNE_URL)
            except Exception as e:
                st.warning(f"Municipality overlay unavailable: {e}")

        if isinstance(muni, dict) and muni.get("features"):
            muni_fields = muni["features"][0].get("properties", {}).keys()
            label_field = "kommnavn" if "kommnavn" in muni_fields else (next(iter(muni_fields)) if muni_fields else None)
            folium.GeoJson(
                muni,
                name="Municipalities",
                style_function=lambda _f: {"fillOpacity": 0.0, "color": "#2c7fb8", "weight": 0.7},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=[label_field] if label_field else [],
                    aliases=["Municipality"] if label_field else [],
                ),
            ).add_to(m)

    # نقاط النقر المحفوظة
    for lat, lon in st.session_state["clicks"]:
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="#ff6b00",
            fill=True, fill_opacity=0.9,
            popup=f"Clicked: {lat:.5f}, {lon:.5f}",
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    # قفل الحدود بالـJS لالتصاق أقوى على الحواف
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

    # عرض الخريطة (تكبير مسموح، تصغير ممنوع تحت MIN_ZOOM)
    event = st_folium(m, height=820, width=1400)  # عدّل الارتفاع إذا رغبت
   

    # Legend ثابت تحت الخريطة
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

    # حفظ آخر نقرة وتحديد منطقتها
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

    if st.session_state["clicks"]:
        st.markdown("**Clicked coordinates (session):**")
        st.dataframe(pd.DataFrame(st.session_state["clicks"], columns=["lat", "lon"]))
    else:
        st.caption("No clicks yet — click on the map to add a pin.")
