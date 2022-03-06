import os
import glob 

import shutil

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
    "Seine_SeineOise",
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


geojson_base_path = os.path.join(
    os.getcwd(), "decomp-set-sum/french_yield/france-geojson/departements/"
)

geojson_folders = glob.glob(geojson_base_path + "/*")

# print(geojson_folders)
dataset_path = os.path.join(os.getcwd(), "french_dept_data")


for dept in depts:
    dept_path = os.path.join(dataset_path, dept)
    os.mkdir(dept_path)

    for year in years:
        os.mkdir(os.path.join(dept_path, year))

    for geo in geojson_folders:
        if dept.lower().replace("_", "-") in geo:
            print(geo, "IN",  dept)
            dept_geojson_file = glob.glob(geo + "/departement*.geojson")[0]
            print(dept_geojson_file)
            shutil.copyfile(dept_geojson_file , dept_path+ "/" + dept + ".geojson")

