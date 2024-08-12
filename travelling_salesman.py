import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
import random
from math import sqrt, exp
import numpy as np
import concurrent.futures
import time


@dataclass
class City:
    id: int
    name: str
    coox: int
    cooy: int


@dataclass
class Arrete:
    phero: float
    ville1: City
    ville2: City
    gamma: float

    def __post_init__(self):
        self.longueur = sqrt((self.ville1.coox - self.ville2.coox) ** 2 + (self.ville1.cooy - self.ville2.cooy) ** 2)

    def set_phero(self, fourmi):
        self.phero = 1

    def evaporation(self):
        self.phero = self.phero * (1 - self.gamma)


@dataclass
class Ant:
    liste_tabou: list[City]
    liste_villes: list[City]
    tbl_arretes: list[list[Arrete]]
    ville_depart: City
    ville_actuelle: City
    ville_suivante: City
    alpha: float
    beta: float

    def __post_init__(self):
        self.liste_probas = []

    def add_liste_tabou(self, ville):
        self.liste_tabou.append(self.ville_actuelle)

    def proba(self, ville):
        if ville == 0:
            return 0
        else:
            if ville != self.ville_actuelle:
                arrete = self.tbl_arretes[self.ville_actuelle.id][ville.id]
                return arrete.phero ** self.alpha * (1 / arrete.longueur) ** self.beta

    def faire_un_pas(self):

        self.liste_tabou.append(self.ville_actuelle)

        if len(self.liste_tabou) == len(self.liste_villes):
            self.liste_tabou.append(self.ville_depart)
            return
        if len(self.liste_tabou) > len(self.liste_villes):
            return

        self.liste_probas = []
        for ville in self.liste_villes:  ## On définit les probas d'aller à telle ou telle ville en partant de celle où on est
            if ville in self.liste_tabou:
                self.liste_probas.append(0)
            else:
                self.liste_probas.append(self.proba(ville))

        self.liste_probas_norm = [float(i) / max(self.liste_probas) for i in self.liste_probas]

        self.ville_actuelle = random.choices(self.liste_villes, weights=self.liste_probas_norm, k=1)[0]

        return

    def calcul_phero(self, q):
        longueur_parcourue = 0
        for k in range(1, len(self.liste_tabou)):
            longueur_parcourue += Arrete(1, self.liste_tabou[k - 1], self.liste_tabou[k], 0.4).longueur

        return q / longueur_parcourue


@dataclass
class Civilisation:
    liste_fourmis: list[Ant]


## Programme ACO :
def ACO(nombre_it, nombre_fourmis, liste_villes, gamma, alpha, beta, seuil_stagnation):
    stagnation_score = 0
    print(f"Nombre d'itérations : {nombre_it}, \nNombre de fourmis : {nombre_fourmis}")
    t1 = time.time()
    total_distance = 0
    tbl_arretes = [[Arrete(1, ville1, ville2, 1) for ville1 in liste_villes] for ville2 in liste_villes]

    for ligne in tbl_arretes:
        for arrete in ligne:
            arrete.gamma = gamma
    liste_villes.sort(key=lambda x: x.id)

    distance_precedente = float('inf')
    stagnation_count = 0

    for k in range(nombre_it):
        # ville_depart = liste_villes[0]
        ville_depart = random.choice(liste_villes)
        liste_fourmis = [Ant([], liste_villes, tbl_arretes, ville_depart, ville_depart, None, alpha, beta) for k in
                         range(nombre_fourmis)]

        for ligne in tbl_arretes:
            for arrete in ligne:
                arrete.evaporation()

        for _ in range(len(liste_villes) + 1):
            for i, fourmi in enumerate(liste_fourmis):
                fourmi.faire_un_pas()

        for fourmi in liste_fourmis:  ## obj : attribuer la quantité de phéromone donnée par chaque fourmi aux arretes concernées
            phero = fourmi.calcul_phero(20)

            for j in range(len(fourmi.liste_tabou) - 1):
                arrete = tbl_arretes[fourmi.liste_tabou[j].id][fourmi.liste_tabou[j + 1].id]
                arrete_bis = tbl_arretes[fourmi.liste_tabou[j + 1].id][fourmi.liste_tabou[j].id]

                arrete.phero += phero
                arrete_bis.phero += phero

                total_distance += arrete.longueur

        # Vérification de la stagnation
        if total_distance >= distance_precedente * 0.98 and total_distance <= distance_precedente * 1.02:
            stagnation_count += 1
            stagnation_score = k/(10*seuil_stagnation)
        else:
            stagnation_count = 0
            stagnation_score = k/(10*seuil_stagnation)

        distance_precedente = total_distance

        # Condition d'arrêt si la stagnation se produit pendant un certain nombre d'itérations consécutives
        if stagnation_count >= seuil_stagnation:
            print(f"Arrêt de l'algorithme en raison de la stagnation pendant {seuil_stagnation} itérations. Nombre d'itérantions effectuées : {k}")
            stagnation_score = - ((nombre_it - k) / (nombre_it))**2 +1
            break
    
    print(stagnation_score)
    infos_graph = (liste_villes, tbl_arretes)
    temps = time.time() - t1
    

    print(f"Temps d'exécution : {temps} secondes")
    return infos_graph, total_distance, stagnation_score


def afficher_graphique_tk(infos_graph):
    liste_villes, tbl_arretes = infos_graph

    fig = plt.figure(figsize=(8, 8))

    # Afficher les villes
    for ville in liste_villes:
        plt.plot(ville.coox, ville.cooy, 'bo')
        plt.text(ville.coox, ville.cooy, ville.name)

    # Coloration des arêtes en fonction du niveau de phéromones
    max_phero = max(arrete.phero for ligne in tbl_arretes for arrete in ligne)
    for ligne in tbl_arretes:
        for arrete in ligne:
            plt.plot([arrete.ville1.coox, arrete.ville2.coox], [arrete.ville1.cooy, arrete.ville2.cooy], 'b-',
                     alpha=(arrete.phero / max_phero) ** 1.5)

    plt.title('Représentation graphique du TSP avec ACO')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Ajouter le graphique matplotlib dans la fenêtre tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=2, rowspan=8)  # Afficher le graphique à côté des entrées


def evaluate(individual, liste_villes, tbl_arretes):
    
    nombre_it = individual['nombre_it']
    nombre_fourmis = individual['nombre_fourmis']
    gamma = individual['gamma']
    alpha = individual['alpha']
    beta = individual['beta']
    
    total_distance, stagnation_score = ACO(nombre_it, nombre_fourmis, liste_villes, gamma, alpha, beta, 20)[1:3]

    fitness = 1 / total_distance * stagnation_score  # Plus la longueur est courte, meilleure est la fitness mais on cherche aussi à avoir un algorithme rapide !
    individual['fitness'] = fitness
    print(f"Fitness : {fitness}\n")
    return individual['fitness']


# Fonction de création d'une population initiale
def create_population(population_size):
    population = []
    for _ in range(population_size):
        individual = {
            'nombre_it': random.randint(30, 120),
            'nombre_fourmis': random.randint(40, 70),
            'gamma': random.uniform(0.01, 0.7),
            'alpha': random.uniform(1, 2.0),
            'beta': random.uniform(0.5, 2.0)
        }
        population.append(individual)
    return population

# Fonction de sélection des individus pour la reproduction
def selection(population, num_parents):
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_population[:num_parents]

# Fonction de croisement (crossover)
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        child = {
            'nombre_it': random.choice([parent1['nombre_it'], parent2['nombre_it']]),
            'nombre_fourmis': random.choice([parent1['nombre_fourmis'], parent2['nombre_fourmis']]),
            'gamma': random.choice([parent1['gamma'], parent2['gamma']]),
            'alpha': random.choice([parent1['alpha'], parent2['alpha']]),
            'beta': random.choice([parent1['beta'], parent2['beta']])
        }
        offspring.append(child)
    return offspring

# Fonction de mutation
def mutation(offspring):
    for individual in offspring:
        if np.random.uniform(0, 1) < 0.2:  # Probabilité de mutation de 20%
            mutation_param = random.choice(['nombre_it', 'nombre_fourmis', 'gamma', 'alpha', 'beta'])
            if mutation_param == 'nombre_it':
                individual['nombre_it'] = random.randint(30, 120)
            elif mutation_param == 'nombre_fourmis':
                individual['nombre_fourmis'] = random.randint(40, 70)
            elif mutation_param == 'gamma':
                individual['gamma'] = random.uniform(0.01, 0.7)
            elif mutation_param == 'alpha':
                individual['alpha'] = random.uniform(1, 2.0)
            elif mutation_param == 'beta':
                individual['beta'] = random.uniform(0.5, 2.0)
    return offspring

# Algorithme génétique principal
def genetic_algorithm(population_size, num_generations, liste_villes, tbl_arretes, progress_callback=None):
    population = create_population(population_size)
    print(f"Generation {1}/{num_generations}")
    
    for k,individual in enumerate(population):
        if progress_callback:
            progress_callback(f"Generation {1}/{num_generations}", f"Individu {k+1}/{len(population)}")
        individual['fitness'] = evaluate(individual, liste_villes, tbl_arretes)  # Calculate fitness for each individual
    
    for gen in range(num_generations-1):
        print(f"Generation {gen+2}/{num_generations}")
        
        parents = selection(population, 2)
        offspring = crossover(parents, population_size - 2)
        offspring = mutation(offspring)
        population = parents + offspring
        
        for k, individual in enumerate(population):
            if progress_callback:
                progress_callback(f"Génération {gen+2}/{num_generations}", f"Individu {k+1}/{len(population)}")
            individual['fitness'] = evaluate(individual, liste_villes, tbl_arretes)
    
    best_individual = max(population, key=lambda x: x['fitness'])
    if progress_callback:
        progress_callback("Algorithme génétique terminé.", "")
    return best_individual


def execute_aco():
    # Vos paramètres
    nombre_it = int(entry_nombre_it.get())
    nombre_fourmis = int(entry_nombre_fourmis.get())
    gamma = float(entry_gamma.get())
    alpha = float(entry_alpha.get())
    beta = float(entry_beta.get())

    # Exécution de l'ACO
    infos_graph = ACO(nombre_it, nombre_fourmis, liste_villes, gamma, alpha, beta, 20)[0]
    root.after(0, afficher_graphique_tk, infos_graph)


def run_genetic_algorithm():
    # Créez un label pour afficher la progression
    progress_label = tk.Label(root, text="Progression de l'algorithme génétique :", pady=10)
    progress_label.grid(row=7, column=0, columnspan=2)
    progress_label1 = tk.Label(root, text="", pady=10)
    progress_label1.grid(row=8, column=0, columnspan=2)
    progress_label2 = tk.Label(root, text="", pady=10)
    progress_label2.grid(row=9, column=0, columnspan=2)

    def update_progress(progress1, progress2):
        progress_label1.config(text=progress1)
        progress_label2.config(text=progress2)
    # Définissez une fonction pour mettre à jour la progression
    def update_progress_wrapper(progress1, progress2):
        root.after(0, update_progress, progress1, progress2)

    # Exécution de l'algorithme génétique pour trouver les meilleurs hyperparamètres
    def run_genetic():
        nonlocal progress_label
        nonlocal update_progress_wrapper

        def update_progress_callback(progress1, progress2):
            update_progress_wrapper(progress1, progress2)

        best_params = genetic_algorithm(population_size=30, num_generations=3, liste_villes=liste_villes, tbl_arretes=tbl_arretes, progress_callback=update_progress_callback)

        # Affichage des meilleurs hyperparamètres trouvés
        messagebox.showinfo("Meilleurs hyperparamètres trouvés", f"Meilleurs hyperparamètres trouvés : {best_params}")
        
        # Mettre à jour les champs d'entrée avec les meilleurs paramètres trouvés
        entry_nombre_it.delete(0, tk.END)
        entry_nombre_it.insert(0, str(best_params['nombre_it']))
        
        entry_nombre_fourmis.delete(0, tk.END)
        entry_nombre_fourmis.insert(0, str(best_params['nombre_fourmis']))
        
        entry_gamma.delete(0, tk.END)
        entry_gamma.insert(0, str(best_params['gamma']))
        
        entry_alpha.delete(0, tk.END)
        entry_alpha.insert(0, str(best_params['alpha']))
        
        entry_beta.delete(0, tk.END)
        entry_beta.insert(0, str(best_params['beta']))

        # Exécuter automatiquement l'ACO avec les nouveaux paramètres
        execute_aco()

        progress_label.destroy()  # Supprimer le label de progression après la fin de l'algorithme

    # Démarrer l'algorithme génétique dans un thread séparé pour ne pas bloquer l'interface
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(run_genetic)




# Création de la fenêtre principale
root = tk.Tk()
root.title("ACO TSP Solver")

# Création des widgets
label_nombre_it = tk.Label(root, text="Nombre d'itérations :")
label_nombre_it.grid(row=0, column=0)
entry_nombre_it = ttk.Entry(root)
entry_nombre_it.grid(row=0, column=1)

label_nombre_fourmis = tk.Label(root, text="Nombre de fourmis :")
label_nombre_fourmis.grid(row=1, column=0)
entry_nombre_fourmis = ttk.Entry(root)
entry_nombre_fourmis.grid(row=1, column=1)

label_gamma = tk.Label(root, text="Gamma :")
label_gamma.grid(row=2, column=0)
entry_gamma = ttk.Entry(root)
entry_gamma.grid(row=2, column=1)

label_alpha = tk.Label(root, text="Alpha :")
label_alpha.grid(row=3, column=0)
entry_alpha = ttk.Entry(root)
entry_alpha.grid(row=3, column=1)

label_beta = tk.Label(root, text="Beta :")
label_beta.grid(row=4, column=0)
entry_beta = ttk.Entry(root)
entry_beta.grid(row=4, column=1)

button_aco = ttk.Button(root, text="Exécuter ACO", command=execute_aco)
button_aco.grid(row=5, column=0, columnspan=2)

button_genetic_algorithm = ttk.Button(root, text="Exécuter Algorithme Génétique", command=run_genetic_algorithm)
button_genetic_algorithm.grid(row=6, column=0, columnspan=2)

# Liste de villes
liste_villes = [
    City(0, "Paris", 50, 75),
    City(1, "Lille", 60, 95),
    City(2, "Lyon", 80, 40),
    City(3, "Bordeaux", 15, 30),
    City(4, "Avignon", 60, 10),
    City(5, "Montpellier", 60, 20),
    City(6, "Nantes", 15, 60),
    City(7, "Brest", 5, 70),
    City(8, "Strasbourg", 90, 70),
    City(9, "Marseille", 65, 5),
    City(10, "Toulouse", 20, 20),
    City(11, "Nice", 75, 5),
    City(12, "Rennes", 20, 65),
    City(13, "Grenoble", 80, 50),
    City(14, "Toulon", 65, 10),
    City(15, "Angers", 25, 60),
    City(16, "Dijon", 70, 60),
    City(17, "Le Havre", 35, 85),
    City(18, "Reims", 60, 80),
    City(19, "Saint-Étienne", 80, 30),
    City(20, "Clermont-Ferrand", 75, 40),
    City(21, "Nîmes", 50, 15),
    City(22, "Limoges", 35, 50),
    City(23, "Tours", 30, 70),
    City(24, "Amiens", 55, 90),
    City(25, "Perpignan", 45, 5),
    City(26, "Besançon", 80, 70),
    City(27, "Orléans", 35, 70),
    City(28, "Metz", 85, 75),
    City(29, "Bourges", 50, 55),
    City(30, "Clermont", 70, 45),
    City(31, "Châteauroux", 45, 55),
    City(32, "Nevers", 50, 45),
    City(33, "Auxerre", 55, 65),

]


liste_villes_courte = [
    City(0, "Paris", 50, 75),
    City(1, "Lille", 60, 95),
    City(2, "Lyon", 80, 40),
    City(3, "Bordeaux", 15, 30),
    City(4, "Avignon", 60, 10),
    City(5, "Montpellier", 60, 20),
    City(6, "Nantes", 15, 60),
    City(7, "Brest", 5, 70),
    City(8, "Strasbourg", 90, 70),
    City(9, "Marseille", 65, 5),
    City(10, "Toulouse", 20, 20),
    City(11, "Nice", 75, 5),
    City(12, "Rennes", 20, 65),
    City(13, "Grenoble", 80, 50),
    City(14, "Toulon", 65, 10),
    City(15, "Angers", 25, 60),
    City(16, "Dijon", 70, 60),
    City(17, "Bourges", 50, 55),
    City(18, "Reims", 60, 80),
    City(19, "Saint-Étienne", 80, 30),
    City(20, "Clermont-Ferrand", 75, 40),
    City(21, "Nîmes", 50, 15),
    City(22, "Limoges", 35, 50),
    City(23, "Tours", 30, 70),
]

# On peut choisir entre deux listes de villes, une longue et une plus courte
liste_villes = liste_villes_courte

tbl_arretes = [[Arrete(1, ville1, ville2, 1) for ville1 in liste_villes] for ville2 in liste_villes]

# Exécution de l'interface
root.mainloop()
