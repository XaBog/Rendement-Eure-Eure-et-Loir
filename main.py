import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("data_sorted.xlsx")

agreste_total = pd.read_excel("agreste_clean_excel.xlsx")

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

era5_eure = era5[era5['latitude'] == 49]
era5_eure = era5_eure[era5_eure['longitude'] == 1]

#era5_eure.to_excel("era5_eure.xlsx")

#precipitations_eure = era5_eure['precipitation']
# print(len(precipitations_eure))
# plt.plot([i for i in range(len(precipitations_eure))], precipitations_eure)
# plt.xlabel("Jour (début 1er janvier 2000)")
# plt.ylabel("Précipitations en 49N 1E")
# plt.show()

#Grouper les précipitations par année, pour comparer au rendement 

# print(era5_eure.columns)

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

#Columns are ['latitude', 'longitude', 'date','precipitatio n', 'r_min', 'ssrd_mean',
       #'Tmax', 'Tavg', 'Tmin', 'ws10_mean']

"""Ce dictionnaire contient les catégories qui seront extraites dans le fichier era5. 
L'utilisateur peut en supprimer une partie si besoin, comme dans l'exemple commenté en dessous, où l'on ne s'intéresse qu'aux précipitations et à la température moyenne"""
labels = {"precipitation" : "Moyenne des précipitations sur l'année", "Tavg" : "Température", "r_min" : "R_min", "ssrd_mean" : "ssrd mean",
       "Tmax" : "Température maximale", "Tmin" : "Température minimale", "ws10_mean" : "ws10_mean"}
#labels = {"precipitation" : "Moyenne des précipitations sur l'année", "Tavg" : "Température"}

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
    value_over_time = value_over_time / mean_value
    plt.plot(rendements_eure['year'], value_over_time, label = labels[value_name], color = "orange")
    plt.plot(rendements_eure['year'], data_rendements_eure_normalisee, label = "Rendement", color = "blue")
    plt.xlabel("Année")
    ylabel = "Valeurs, unités arbitraires"
    plt.ylabel(ylabel)
    plt.legend()
    save_path = "courbes/" + labels[value_name]
    #plt.savefig(save_path)
    plt.close()
    return value_over_time

# for column in labels.keys():
#     compare_value_to_rendement(column)

def nombre_de_jours_de_gel_par_an():
    """Cette fonction calcule le nombre de jours où la température minimale dans l'Eure a été en dessous de 0 degrés
    Elle renvoie un tableau 1D donnant le nombre moyen de jours de gel par an chaque année, normalisé pour que la moyenne de ce tableau soit 1. 
    Elle affiche également la comparaison de ce tableau avec le rendement par an"""
    temperatures_minimales = era5_eure['Tmin']
    nyears = 21
    resultats = [0 for year in range(nyears)]
    for year in range(nyears):
        for day in range(365):
            if temperatures_minimales.iloc[365*year + day] < 0:
                resultats[year] += 1
    resultats.append(resultats[20])
    resultats = np.array(resultats)
    resultats = resultats/np.mean(resultats)
    plt.plot(rendements_eure['year'], resultats, label = "Nombre de jours de gel par an", color = "orange")
    plt.plot(rendements_eure['year'], data_rendements_eure_normalisee, label = "Rendement", color = "blue")
    plt.xlabel("Année")
    ylabel = "Valeurs, unités arbitraires"
    plt.ylabel(ylabel)
    plt.legend()
    save_path = "courbes/" +  "Comparaison rendement vs nombre de jours de gel par an"
    plt.savefig(save_path)
    plt.close()
    return resultats

def nombre_de_jours_de_gel_printemps():
    """Cette fonction calcule le nombre de jours de gel de chaque année entre le 1er janvier et fin juillet, époque des récoltes du blé épeautre. 
    Elle renvoie un tableau 1D donnant le nombre moyen de jours de gel sur cette période chaque année, normalisé pour que la moyenne de ce tableau soit 1. 
    Elle affiche également la comparaison de ce tableau avec le rendement par an"""
    temperatures_minimales = era5_eure['Tmin']
    nyears = 21
    resultats = [0 for year in range(nyears)]
    for year in range(nyears):
        for day in range(210): #La récolte se fait en juillet, donc seul le gel au printemps est important
            if temperatures_minimales.iloc[365*year + day] < 0:
                resultats[year] += 1
    resultats.append(resultats[20])
    resultats = np.array(resultats)
    resultats = resultats/np.mean(resultats)
    plt.plot(rendements_eure['year'], resultats, label = "Nombre de jours de gel par an", color = "orange")
    plt.plot(rendements_eure['year'], data_rendements_eure_normalisee, label = "Rendement", color = "blue")
    plt.xlabel("Année")
    ylabel = "Valeurs, unités arbitraires"
    plt.ylabel(ylabel)
    plt.legend()
    save_path = "courbes/" +  "Comparaison rendement vs nombre de jours de gel en fin d'hiver printemps"
    plt.savefig(save_path)
    plt.close()
    return resultats

# nombre_de_jours_de_gel_printemps()

def get_rendement_moyen_national_par_an():
    """Cette fonction extrait du fichier agreste le rendement moyen national par an. 
    Il est inutile de l'appeler, le résultat étant déjà sauvegardé dans la suite du code"""
    data_rendements_nationaux = [0 for i in range(21)]
    for i in range(21):
        year = 2000 + i
        rendements_nationaux_cette_annee = agreste_total[agreste_total['year'] == year]
        rendements_nationaux_cette_annee = rendements_nationaux_cette_annee[agreste_total['n6'] == "Blé tendre d'hiver et épeautre"]
        rendements_nationaux_cette_annee = rendements_nationaux_cette_annee[rendements_nationaux_cette_annee['variable'] == 'Rendement']
        rendement_moyen_cette_annee = rendements_nationaux_cette_annee.mean(numeric_only=True)
        data_rendements_nationaux[i] = rendement_moyen_cette_annee['value']
        if i == 0:
            print(rendements_nationaux_cette_annee)
    return (data_rendements_nationaux)

#data_rendements_nationaux = get_rendement_moyen_national_par_an() #results below to avoid making computations multiple times
data_rendements_nationaux = np.array([63.43617021276596, 57.878723404255325, 66.04787234042553, 53.376344086021504, 66.83333333333333, 62.30107526881721, 60.225806451612904, 
                             56.03225806451613, 62.74193548387097, 64.56989247311827, 63.11720430107526, 58.82795698924731, 64.1741935483871, 64.37204301075269, 
                             64.24086021505376, 67.40860215053763, 51.0372340425532, 64.49893617021277, 59.77234042553191, 68.0872340425532, 58.29787234042554])
data_rendements_nationaux_normalises = data_rendements_nationaux/np.mean(data_rendements_nationaux)

# plt.plot([2000 + i for i in range(21)], data_rendements_nationaux_normalises, label = "France")
# plt.plot([2000 + i for i in range(22)], data_rendements_eure_normalisee, label = "Eure")
# plt.legend()
# plt.show()

"""Les 4 lignes ci-dessous extraient du fichier era5 les différentes variables météorologiques qui nous intéresseront par la suite, pour le département de l'Eure"""
values_over_time = []
for column in labels.keys():
    # print(column)
    values_over_time.append(compare_value_to_rendement(column))

def somme_lineaire(coeffs):
    """Cette fonction effectue la somme linéaire des courbes du fichier era5, pondérées par les coefficients donnés en entrée.
    Elle renvoie la courbe correspondant à cette somme"""
    courbe = [0 for i in range(22)]
    for index, value_over_time in enumerate(values_over_time):
        courbe += coeffs[index] * value_over_time
    courbe = courbe/np.mean(courbe)
    return courbe

def erreur(coeffs):
    """Cette fonction calcule et renvoie l'erreur liée au set de coefficients donnés en entrée."""
    courbe = somme_lineaire(coeffs)
    res = 0
    for i in range(22):
        res += (courbe[i] - data_rendements_eure_normalisee.iloc[i])**2
    return res

def monte_carlo_sampling(N_samples = 100000):
    """Cette fonction génère N_samples sets de coefficients pour la somme pondérée des variables du fichier era5.
    Elle renvoie le set de coefficients le plus proche des rendements observés dans le fichier agreste.
    Elle affiche également la courbe correspondante."""
    coefficients_actuels = 10**np.random.uniform(-2, 1, [len(labels.keys())])
    print("Colonnes = ", labels.keys())
    print("coeffs de départ = ", coefficients_actuels)
    meilleure_performance = erreur(coefficients_actuels)
    for i in range(N_samples):
        coefficients = 10**np.random.uniform(-4,  1, [len(labels.keys())])
        perf = erreur(coefficients)
        if erreur(coefficients) < meilleure_performance:
            coefficients_actuels = coefficients
            meilleure_performance = perf
    print("Best coeffs = ", coefficients_actuels)
    plt.plot([i for i in range(22)], somme_lineaire(coefficients_actuels), label = "Meilleure courbe")
    plt.plot([i for i in range(22)], data_rendements_eure_normalisee, label = "Rendement")
    plt.legend()
    plt.show()
    return coefficients_actuels

"""Ce code utilise la fonction ci-dessous puis calcule la corrélation du résultat ; cette corrélation est très faible, avec 10 millions d'échantillons testés."""
# coeffs = monte_carlo_sampling()
# coeffs_correlation = np.corrcoef(somme_lineaire(coeffs), data_rendements_eure_normalisee)
# print(coeffs_correlation)

"""Ce code calcule les corrélations entre les variables du fichier era5 et l'évolution du rendement annuel dans l'Eure. 
La meilleure corrélation est pour la température minimale, mais celle-ci reste très faible"""
# for i in range(len(labels.keys())):
#     coeff = np.corrcoef(data_rendements_eure_normalisee, values_over_time[i])
#     print(coeff)

"""Ce code calcule la corrélation entre l'évolution du rendement national et du rendement dans l'Eure. Celle-ci est satisfaisante (0.87)"""
# data_rendements_nationaux_normalises_agrandie = np.zeros([22])
# for i in range(21):
#     data_rendements_nationaux_normalises_agrandie[i] = data_rendements_nationaux_normalises[i]
# data_rendements_nationaux_normalises_agrandie[21] = data_rendements_nationaux_normalises_agrandie[20]
# print(np.corrcoef(data_rendements_nationaux_normalises_agrandie, data_rendements_eure_normalisee))
      
def extraire_data_eure_sur_un_an(year):
    """Ne fonctionne pas pour l'année 0, puisque l'on veut regarder depuis octobre de l'année n-1 jusqu'aux récoltes"""
    year_index = year - 2000
    print("Rendement cette année : ", data_rendements_eure.iloc[year_index])
    values_over_time = []
    nvalues = len(labels.keys())
    for value_name in labels.keys():
        data_considered = era5_eure[value_name]
        value_over_time = [0 for day in range(365)]
        for day in range(365):
            value_over_time[day] += data_considered.iloc[365*year_index - 90 + day]
        value_over_time = np.array(value_over_time)
        #value_over_time = value_over_time/np.mean(value_over_time)
        values_over_time.append([value_over_time, value_name])
        print("Moyenne cette année : ", np.mean(value_over_time))
    for index, element in enumerate(values_over_time):
        data = element[0]
        label = element[1]
        plt.subplot(nvalues, 1, index + 1)
        plt.plot([i for i in range(365)], data, label = label)
        plt.legend()
    plt.show()

# extraire_data_eure_sur_un_an(2016)

def extraire_data_france_sur_un_an(year):
    print("Rendement pour l'année " + str(year) + " = ", data_rendements_nationaux[year - 2000])
    data_considered = []
    ndays = 365
    if year%4 == 0:
        ndays = 366
    for index, element in era5.iterrows():
        date = element['date']
        if int(date[0:4]) == year and np.random.uniform(0, 1) > 0.1:
            data_considered.append(element)
    data_considered = pd.DataFrame(data_considered)
    data_considered = data_considered.groupby('date').mean() 
    for index, value_name in enumerate(labels.keys()):
        plt.subplot(len(labels.keys()), 1, index + 1)
        values = data_considered[value_name]
        mean_value = values.mean()
        label = labels[value_name]
        print("Valeur moyenne pour " + label + " en " + str(year) + " = ", mean_value)      
        plt.plot([i for i in range(ndays)], values, label = label)
        plt.legend()
    plt.show()
    plt.close()

# extraire_data_france_sur_un_an(2015)
# extraire_data_france_sur_un_an(2016)












    










