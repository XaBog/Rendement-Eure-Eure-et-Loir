import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("data_sorted.xlsx")

data_eure = file[file['dpt'] == 27]
data_loir = file[file['departement'] == 'Eure-et-Loir']

data_eure  = data_eure.sort_values(by = ['year'])
data_loir = data_loir.sort_values(by = ['year'])

# print("Data Eure : ")
# print(data_eure)
# print(" ")
# print("Data Eure-et-Loir :")
# print(data_loir)
# print(" ")

rendements_eure = data_eure[data_eure['variable'] == 'Rendement']
data_rendements_eure = rendements_eure['value']
rendement_moyen = data_rendements_eure.mean(numeric_only=True)
data_rendements_eure_normalisee = data_rendements_eure.divide(rendement_moyen)
rendement_moyen_normalise = 1

# print(rendements_eure)
# print(len(rendements_eure))

# plt.plot(rendements_eure['year'], rendements_eure['value'], label = "Rendement", color = "blue")
# plt.xlabel("Année")
# plt.ylabel("Rendement")
# plt.legend()
# plt.show()
era5 = pd.read_csv("era5.csv")

era5_eure = era5[era5['latitude'] < 49.1]
era5_eure = era5_eure[era5_eure['latitude'] > 48.9]
era5_eure = era5_eure[era5_eure['longitude'] > 0.95]
era5_eure = era5_eure[era5_eure['longitude'] < 1.05]

#era5_eure.to_excel("era5_eure.xlsx")

precipitations_eure = era5_eure['precipitation']

# print(len(precipitations_eure))
# plt.plot([i for i in range(len(precipitations_eure))], precipitations_eure)
# plt.xlabel("Jour (début 1er janvier 2000)")
# plt.ylabel("Précipitations en 49N 1E")
# plt.show()

#Grouper les précipitations par année, pour comparer au rendement 

print(era5_eure.columns)

# nyears = len(rendements_eure)
# print("nyears = ", nyears)
# print(365*nyears)
# precipitations_moyennes = [0 for i in range(21)]
# for i in range(21):
#     for day in range(365):
#         precipitations_moyennes[i] += precipitations_eure.iloc[365*i + day]
#     precipitations_moyennes[i] = precipitations_moyennes[i]/365

# precipitations_moyennes.append(precipitations_moyennes[20])

# precipitations_moyennes = np.array(precipitations_moyennes)
# moyenne_totale = np.mean(precipitations_moyennes)

# precipitations_moyennes = 80*precipitations_moyennes/moyenne_totale

# plt.plot(rendements_eure['year'], precipitations_moyennes, label = "Précipitations")
# plt.plot(rendements_eure['year'], rendements_eure['value'], label = "Rendement")
# plt.xlabel("Année")
# plt.legend()
# plt.show()

#Columns are ['latitude', 'longitude', 'date', 'precipitation', 'r_min', 'ssrd_mean',
       #'Tmax', 'Tavg', 'Tmin', 'ws10_mean']
legends = {"precipitation" : "Moyenne des précipitations sur l'année", "Tavg" : "Température", "r_min" : "R_min", "ssrd_mean" : "ssrd mean",
       "Tmax" : "Température maximale", "Tmin" : "Température minimale", "ws10_mean" : "ws10_mean"}

def compare_value_to_rendement(value_name):
    """Compares the evolution of a value (e.g, precipitations) from era5 to the evolution of the rendement
    Args :
        value_name : string
    Outputs:
        the array of the evolution of said value"""
    nyears = 21
    data_considered = era5_eure[value_name]
    value_over_time = [0 for i in range(nyears)]
    for year in range(nyears):
        for day in range(365):
            value_over_time[year] += data_considered.iloc[365*year + day]
        value_over_time[year] = value_over_time[year]/365
    value_over_time.append(value_over_time[20])
    value_over_time = np.array(value_over_time)
    mean_value = np.mean(value_over_time)
    value_over_time = value_over_time * rendement_moyen_normalise / mean_value
    plt.plot(rendements_eure['year'], value_over_time, label = legends[value_name], color = "orange")
    plt.plot(rendements_eure['year'], data_rendements_eure_normalisee, label = "Rendement", color = "blue")
    plt.xlabel("Année")
    ylabel = "Valeurs, unités arbitraires"
    plt.ylabel(ylabel)
    plt.legend()
    save_path = "courbes/" + legends[value_name]
    plt.savefig(save_path)
    plt.close()

# for column in legends.keys():
#     compare_value_to_rendement(column)

