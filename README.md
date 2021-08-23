
# Solutions to TSP instances of Very-Large-Scale Integration (VLSI) design

This repository describes solutions that can lead to optimal or near-optimal 
combination of MOS transistors onto silicon chips in order to create **integrated
circuits (ICs)**, a process which is called **Very-Large-Scale Integration (VLSI)**. 

## Approach

The process of VLSI was examined by using approaches of the known optimization 
combinatorial problem called **Traveling Salesperson Problem (TSP)** TSP.
TSP is described as follows:
> Given a set of cities and the distance between each pair of cities, what is the shortest possible tour that visits each city exactly once, and returns to the starting city?

The parameters that have been taken into account are the total distance of the process
by starting and ending at one point and visiting all other points exactly once 
and the time of the computation. 

## Datasets

The datasets used in this project were provided by Andre Rohe, 
based on VLSI data sets studied at the Forschungsinstitut für Diskrete Mathematik, 
Universität Bonn [VLSI Instances](http://www.math.uwaterloo.ca/tsp/vlsi/index.html). 
These datasets contain coordinates of points of interest in 2-D. In this repository's notebooks, 
we used instance from dozens of points up to several thousand ones.

## Notebooks

* Optimal_solutions_VLSI.ipynb : Optimal solutions ( Calculation of All tours and Held-Karp algorithm)
* Near_Optimal_Solutions_VLSI.ipynb : Near-Optimal solutions (Calculation of near-optimal tours without guaranteeing finding any near-optimal solutions)
* Minimum_Spanning_Tree_VLSI.ipynb  : Near-Optimal solutions (Calculation of near-optimal tours by guaranteeing finding near-optimal solutions)

## Useful sources

* [Peter Norvig's post about TSP](https://nbviewer.jupyter.org/url/norvig.com/ipython/TSP.ipynb) 
* [ University of Waterloo, Department of Combinatorics and Optimization webpage ](http://www.math.uwaterloo.ca/tsp/) 
* [William Cook's presentation on Youtube about TSP](https://www.youtube.com/watch?v=q8nQTNvCrjE&t=35s) 