import rasterio
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




dataset_path = os.path.join(os.getcwd(), "french_dept_data")

print(months.values())

for dept in depts:
    for year in years:
        dept_year_path = os.path.join(dataset_path, dept, year)
        month_file_names = glob.glob(dept_year_path + "/*.tif")
        # print(month_file_names)
        month_tifs = []
        for month in months.values():
            # print(month)
            for mfn in month_file_names:
                if month in mfn:
                    # print(mfn)
                    month_tifs.append(rasterio.open(mfn))
        
        merged_geo = month_tifs[0].profile
        merged_geo.update({"count": 6})

        merged_name = os.path.join(dept_year_path, f"merged_{dept}_{year}.tif")

        with rasterio.open(merged_name, 'w', **merged_geo) as dest:
            for i, band in enumerate(month_tifs):
                dest.write(band.read(1),i + 1 )




# band2=rasterio.open("B02.jp2")
# band3=rasterio.open("B03.jp2")
# band4=rasterio.open("B04.jp2")

# band2_geo = band2.profile
# band2_geo.update({"count": 3})

# with rasterio.open('rgb.tiff', 'w', **band2_geo) as dest:
# # I rearanged the band order writting to 2→3→4 instead of 4→3→2
#     dest.write(band2.read(1),1)
#     dest.write(band3.read(1),2)
#     dest.write(band4.read(1),3)