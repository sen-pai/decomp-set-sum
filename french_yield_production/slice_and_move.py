import rasterio
from rasterio.mask import mask

import fiona

import os 
import glob

depts = [
    "Ain",
    "Aisne",
    "Allier",
    "Alpes_de_Haute_Provence",
    "Alpes_Maritimes",
    "Ardeche",
    "Ardennes",
    "Ariege",
    "Aube",
    "Aude",
    "Aveyron",
    "Bas_Rhin",
    "Bouches_du_Rhone",
    "Calvados",
    "Cantal",
    "Charente",
    "Charente_Maritime",
    "Cher",
    "Correze",
    "Corse_du_Sud",
    "Cote_d_Or",
    "Cotes_d_Armor",
    "Creuse",
    "Deux_Sevres",
    "Dordogne",
    "Doubs",
    "Drome",
    "Essonne",
    "Eure",
    "Eure_et_Loir",
    "Finistere",
    "Gard",
    "Gers",
    "Gironde",
    "Haut_Rhin",
    "Haute_Corse",
    "Haute_Garonne",
    "Haute_Loire",
    "Haute_Marne",
    "Haute_Saone",
    "Haute_Savoie",
    "Haute_Vienne",
    "Hautes_Alpes",
    "Hautes_pyrenees",
    "Hauts_de_Seine",
    "Herault",
    "Ille_et_Vilaine",
    "Indre",
    "Indre_et_Loire",
    "Isere",
    "Jura",
    "Landes",
    "Loir_et_Cher",
    "Loire",
    "Loire_Atlantique",
    "Loiret",
    "Lot",
    "Lot_et_Garonne",
    "Lozere",
    "Maine_et_Loire",
    "Manche",
    "Marne",
    "Mayenne",
    "Meurthe_et_Moselle",
    "Meuse",
    "Morbihan",
    "Moselle",
    "Nievre",
    "Nord",
    "Oise",
    "Orne",
    "Paris",
    "Pas_de_Calais",
    "Puy_de_Dome",
    "Pyrenees_Atlantiques",
    "Pyrenees_Orientales",
    "Rhone",
    "Saone_et_Loire",
    "Sarthe",
    "Savoie",
    "Seine_et_Marne",
    "Seine_Maritime",
    "Seine_Saint_Denis",
    # "Seine_SeineOise",
    "Somme",
    "Tarn",
    "Tarn_et_Garonne",
    "Territoire_de_Belfort",
    "Val_d_Oise",
    "Val_de_Marne",
    "Var",
    "Vaucluse",
    "Vendee",
    "Vienne",
    "Vosges",
    "Yonne",
    "Yvelines",
]


years = [
    "2002",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
]


months = {
    "0210_006": "feb",
    "0310_006": "mar",
    "0410_006": "apr",
    "0510_006": "may",
    "0610_006": "jun",
    "0710_006": "jul",
    
}


ndvi_data_path = os.path.join(os.getcwd(), "ndvi_data")
dataset_path = os.path.join(os.getcwd(), "french_dept_data")

get_name = lambda year, month: f'MCD13A2_{year}{month}_globalV1_1km_OF/NDVI/ndvi.tif'
get_cut_name = lambda dept, year, month: f'{dept}_{year}_{months[month]}.tif'

for year in years:
    year_data_path = os.path.join(ndvi_data_path, year)
    for month in months.keys():
        ndvi_file_path = os.path.join(year_data_path, get_name(year, month))
        for dept in depts:
            dept_path = os.path.join(dataset_path, dept)
            dept_year_path = os.path.join(dept_path, year)

            geo_json_name = glob.glob(dept_path + "/*.geojson")[0]
            geo_json_path = os.path.join(dataset_path, geo_json_name)

            cut_save_path = os.path.join(dept_year_path, get_cut_name(dept, year, month) )


            with fiona.open(geo_json_path) as shapefile:
                geoms = [feature["geometry"] for feature in shapefile]

            # load the raster, mask it by the polygon and crop it
            with rasterio.open(ndvi_file_path) as src:
                out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()

            out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2 ],
            "transform": out_transform})

            with rasterio.open(cut_save_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            print(f'NDVI file {ndvi_file_path} is cut with {geo_json_name} saved at {cut_save_path} ')
