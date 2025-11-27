import json
import geopandas as gpd
import plotly.express as px

url = "https://nve.geodataonline.no/arcgis/rest/services/Mapservices/Elspot/MapServer/0/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
gdf = gpd.read_file(url)

fig = px.choropleth_map(
    gdf,
    geojson=json.loads(gdf.to_json()),
    locations=gdf.index,
    color="ElSpotOmr",
    hover_name="ElSpotOmr",
    center={"lat": 64.5, "lon": 11.0},
    zoom=4.5,
    opacity=0.6,
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_html("nve_elspot_plotly.html")
print("✅ Map saved as nve_elspot_plotly.html")
