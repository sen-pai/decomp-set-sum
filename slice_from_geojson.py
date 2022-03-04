import rasterio
from rasterio.mask import mask
# the polygon GeoJSON geometry
geoms = [{'type': 'Polygon', 'coordinates': [[(250204.0, 141868.0), (250942.0, 141868.0), (250942.0, 141208.0), (250204.0, 141208.0), (250204.0, 141868.0)]]}]
# load the raster, mask it by the polygon and crop it
with rasterio.open("test.tif") as src:
    out_image, out_transform = mask(src, geoms, crop=True)
out_meta = src.meta.copy()

# save the resulting raster  
out_meta.update({"driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
"transform": out_transform})

with rasterio.open("masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)