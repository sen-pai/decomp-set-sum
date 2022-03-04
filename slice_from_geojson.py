import rasterio
from rasterio.mask import mask

import fiona

with fiona.open('french_yield/france-geojson/departements/02-aisne/departement-02-aisne.geojson') as shapefile:
    geoms = [feature["geometry"] for feature in shapefile]


print(geoms)
# # the polygon GeoJSON geometry
# with open('french_yield/france-geojson/departements/02-aisne/departement-02-aisne.geojson') as data_file:    
#     geoms= json.load(data_file)

print("geom loaded")

# load the raster, mask it by the polygon and crop it
with rasterio.open("ndvi.tif") as src:
    out_image, out_transform = mask(src, geoms, crop=True)
out_meta = src.meta.copy()

print("original tif loaded")

# save the resulting raster  
out_meta.update({"driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
"transform": out_transform})

with rasterio.open("aisne_ndvi_masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)