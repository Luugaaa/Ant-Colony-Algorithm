# ACO TSP Solver with Genetic Algorithm Optimization

This project implements an Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP), with a feature to optimize the ACO parameters using a Genetic Algorithm (GA). The project includes a graphical interface built with Tkinter to visualize the results.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [ACO Implementation for TSP](#aco-implementation-for-tsp)
   - [ACO Hyperparameters Selection and Impact](#aco-hyperparameters-selection-and-impact)
   - [Genetic Algorithm for Hyperparameter Optimization](#genetic-algorithm-for-hyperparameter-optimization)
   - [Fitness Function](#fitness-function)
3. [Code Description](#code-description)
   - [Data Structures](#data-structures)
   - [Main Functions](#main-functions)
   - [Tkinter Interface](#tkinter-interface)
4. [Experimental Results](#experimental-results)
   - [Performance Analysis of ACO with GA Optimization](#performance-analysis-of-aco-with-ga-optimization)
   - [Performance Comparison](#performance-comparison)
5. [Discussion](#discussion)
   - [Interpretation of Results](#interpretation-of-results)
   - [Advantages and Limitations](#advantages-and-limitations)
   - [Future Improvements](#future-improvements)
6. [Conclusion](#conclusion)

## Introduction

The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem that seeks the shortest possible route visiting each city in a set exactly once and returning to the starting city. This project aims to develop a solution using Ant Colony Optimization (ACO), a metaheuristic inspired by the foraging behavior of ants. 

To enhance the efficiency of ACO, a Genetic Algorithm (GA) is employed to automatically optimize its hyperparameters, such as the number of iterations, the number of ants, and pheromone influence coefficients. This combination aims to find high-quality solutions within a reasonable time frame, with results visualized via a Tkinter-based GUI.

## Methodology

### ACO Implementation for TSP

The ACO algorithm simulates the process of ants traversing a graph of cities, depositing and following pheromones on the edges. The main components include:
- **City Class:** Represents a node in the TSP graph with unique identifiers and coordinates.
- **Arrete Class:** Represents an edge between cities, containing information about pheromone levels and distances.
- **Ant Class:** Represents an individual ant constructing a solution, maintaining a list of visited cities and pheromone influence parameters.
- **Civilisation Class:** Represents a group of ants working together in the ACO.

### ACO Hyperparameters Selection and Impact

The performance of ACO heavily relies on its hyperparameters. Key parameters include the number of iterations, the number of ants, the pheromone evaporation rate, and the influence coefficients. Proper tuning of these parameters is crucial for the algorithm’s success in finding high-quality solutions efficiently.

### Genetic Algorithm for Hyperparameter Optimization

To automate the selection of optimal hyperparameters, a Genetic Algorithm (GA) is implemented. The GA mimics natural selection by evolving a population of potential solutions over several generations. The best-performing hyperparameters identified by the GA are then used to execute the ACO.

### Fitness Function

The fitness of a solution is evaluated based on the total path length traversed by the ants, adjusted by a stagnation score to account for the algorithm's convergence speed. The fitness function encourages exploration by penalizing excessive stagnation, ensuring that the ACO continues to find better solutions over time.

## Code Description

### Data Structures

Several classes are defined to model the TSP and the ACO:
- **City:** Represents a city in the TSP, characterized by a unique ID, name, and Cartesian coordinates.
- **Arrete:** Represents an edge in the TSP graph, storing information such as pheromone levels and edge length.
- **Ant:** Models an ant in the ACO algorithm, maintaining a list of visited cities and relevant parameters.
- **Civilisation:** Represents a colony of ants working together to solve the TSP.

### Main Functions

- **ACO Function:** Implements the ACO algorithm, taking parameters like the number of iterations, ants, and ACO hyperparameters.
- **Evaluate Function:** Assesses the fitness of a set of ACO hyperparameters by running the ACO and measuring the quality of the resulting solution.
- **Genetic Algorithm Function:** Implements the GA to optimize the ACO hyperparameters, evolving a population of potential solutions over several generations.

### Tkinter Interface

The GUI, built with Tkinter, allows users to interact with the algorithm by setting ACO parameters, running the algorithm, and visualizing the results. It also includes functionality to run the GA for automatic hyperparameter optimization.

## Experimental Results

### Performance Analysis of ACO with GA Optimization

Experiments demonstrate significant improvements in ACO performance following GA-based hyperparameter optimization. The optimized ACO produces shorter paths more quickly, indicating better solution quality and efficiency.

### Performance Comparison

Comparing pre- and post-optimization results shows a marked reduction in the path lengths and execution times. The GA efficiently finds the minimal conditions necessary for the best results, thereby reducing computational overhead.

## Discussion

### Interpretation of Results

The results confirm that GA optimization significantly enhances ACO's performance, with optimized hyperparameters leading to better-quality solutions and faster convergence.

### Advantages and Limitations

The combined ACO-GA approach effectively solves the TSP by finding high-quality solutions. However, GA optimization can be computationally intensive, especially for large problem instances.

### Future Improvements

Future work could explore alternative hyperparameter optimization techniques, such as evolutionary algorithms or Bayesian optimization. Additionally, hybrid metaheuristics combining ACO with other optimization methods might yield even better results.

## Conclusion

This project demonstrates the successful application of ACO to solve the TSP, with GA-based hyperparameter optimization significantly improving its performance. The approach offers practical benefits for combinatorial optimization problems and opens avenues for future research in metaheuristic optimization.
