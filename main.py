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
# print(rendements_eure)
# print(len(rendements_eure))

# plt.plot(rendements_eure['year'], rendements_eure['value'])
# plt.xlabel("Année")
# plt.ylabel("Rendement")
# plt.show()

era5 = pd.read_parquet("ERA5_data.parquet")

precip = era5['precipitation']
print(type(precip))
