import rasterio
import os
import glob
import shutil

from itertools import product
import rasterio as rio
from rasterio import windows


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




def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


dataset_path = os.path.join(os.getcwd(), "../french_dept_data")


for dept in depts:
    for year in years:
        dept_year_path = os.path.join(dataset_path, dept, year)
        merged_input_name = glob.glob(dept_year_path + "/merged*.tif")[0]
        output_folder_name = f"split_{dept}_{year}_1"
        output_folder_path = os.path.join(dept_year_path, output_folder_name)

        if os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
        os.mkdir(output_folder_path)

        output_filename = 'subtile_{}-{}.tif'


        with rio.open(merged_input_name) as inds:
            tile_width, tile_height = 1, 1
            meta = inds.meta.copy()

            for window, transform in get_tiles(inds, tile_width, tile_height):
                print(window)
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(output_folder_path, output_filename.format(int(window.col_off), int(window.row_off)))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))


