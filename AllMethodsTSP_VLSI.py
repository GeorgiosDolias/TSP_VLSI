#Import necessary packages

#% matplotlib inline
import matplotlib.pyplot as plt
from itertools import permutations
import functools
from pandas import read_csv
from math import sqrt
from time import perf_counter
import random


class AllMethodsTSP_VLSI():
    
    tsp_algorithm = ""
        
    def __init__(self, dataframe,tsp_algorithm= None):
        self.cities = frozenset(zip(dataframe['X'],dataframe['Y']))
        #if condition returns False, AssertionError is raised
        assert len(self.cities) != 0, "Coordinates weren't imported correctly. Please try again."
        self.tsp_algorithm = tsp_algorithm
    
    def TSP_solution(self, repetitions=None):
        self.plot_tsp(self.tsp_algorithm,self.cities, repetitions)    
    
    
    def alltours_tsp(self,cities):
        "Generate all possible tours of the cities and choose the shortest tour."
        
        return self.shortest_tour(self.alltours(self.cities))
    
    def alltours(self, cities):
        "Return a list of tours, each a permutation of cities, but each one starting with the same city."
        start = self.first(cities)
        return [[start] + list(rest) for rest in permutations(cities - {start})]
    
    def first(self,collection):
        "Start iterating over collection, and return the first element."
        return next(iter(collection))
    
    def shortest_tour(self, tours):
        "Choose the tour with the minimum tour length."
        return min(tours, key=self.tour_length)
    
    def tour_length(self, tour):
        "The total of distances between each pair of consecutive cities in the tour."
        return sum(self.distance(tour[i], tour[i-1]) for i in range(len(tour)))
        
    def distance(self, A, B):
        """Calculate distance between two points"""
        return sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)
    
    def plot_tsp(self, algorithm, cities,repetitions=None):
        "Apply a TSP algorithm to cities, plot the resulting tour, and print information."
        # Find how long the solution and time take
        t0 = perf_counter()
        
        if algorithm == "repeat_rep_nn_tsp":
            tour = getattr(self,algorithm)(self.cities, repetitions)
        else:    
            tour = getattr(self,algorithm)(self.cities)
        
        t1 = perf_counter()
        assert self.valid_tour(tour, cities)
        
        if algorithm in ["greedy_tsp","nn_tsp", "altered_greedy_tsp"]:
            self.plot_tour(tour,algorithm)
        else:
            self.plot_tour(tour)
        plt.show()
        print("{} city tour with length {:.3f} in {:.3f} secs for {}"
              .format(len(tour), self.tour_length(tour), t1 - t0, getattr(self,algorithm).__name__))
    
    def valid_tour(self, tour, cities):
        "Is tour a valid tour for these cities?"
        return len(tour) == len(cities)
    
    def plot_tour(self, tour,algorithm = None): 
        "Plot the cities as circles and the tour as lines between them."
        if algorithm in ["greedy_tsp","nn_tsp", "altered_greedy_tsp"]:
            "Plot the cities as circles and the tour as lines between them."
            start = tour[0]
            plt.figure(figsize=(10, 10))
            self.plot_lines(list(tour) + [tour[0]])
            self.plot_lines([start], 'rs') # Mark the start city with a red square
        else:
            self.plot_lines(list(tour) + [tour[0]])
    
    def plot_lines(self, points, style='go-'):
        "Plot lines to connect a series of points."
        if self.tsp_algorithm not in ["nn_tsp","greedy_tsp", "altered_greedy_tsp"]:
            plt.figure(figsize=(10, 10))
        plt.plot([p[0] for p in points], [p[1] for p in points], style)
        plt.axis('scaled'); plt.axis('off')
    
    
    def hk_tsp(self,cities):
        """The Held-Karp shortest tour of this set of cities.
        For each end city C, find the shortest segment from A (the start) to C.
        Out of all these shortest segments, pick the one that is the shortest tour."""
        A = self.first(self.cities)
        return self.shortest_tour(self.shortest_segment(A, cities - {A, C}, C)
                             for C in self.cities if C is not A)
    
    @functools.lru_cache(None)
    def shortest_segment(self,A, Bs, C):
        "The shortest segment starting at A, going through all Bs, and ending at C."
        if not Bs:
            return [A, C]
        else:
            segments = [self.shortest_segment(A, Bs - {B}, B) + [C] 
                        for B in Bs]
            return min(segments, key=self.segment_length)

    def segment_length(self, segment):
        "The total of distances between each pair of consecutive cities in the segment."
        return sum(self.distance(segment[i], segment[i-1]) 
                   for i in range(1, len(segment)))
    
    def benchmarks(self, tsp_algorithms, cities):
        "Print benchmark statistics for each of the algorithms."    
        for tsp in tsp_algorithms:
            time, results = self.benchmark(getattr(self,tsp), self.cities)
            lengths = self.tour_length(results) 
            print("Algorithm: {:>15}, executed in: {:7.8f} secs for: {:3.0f} cities, with total tour length: {:5.3f} "
                  .format(getattr(self,tsp).__name__,  time, len(self.cities), lengths))
    
    @functools.lru_cache(None)
    def benchmark(self,function, cities):
        "Run function on all the inputs; return pair of (average_time_taken, results)."
        t0           = perf_counter()
        results      = function(self.cities)
        t1           = perf_counter()
        execution_time = t1 - t0
        return (execution_time, results)  
    
    
    def nn_tsp(self,cities,start=None):
        """Start the tour at the first city; at each step extend the tour 
        by moving from the previous city to the nearest neighboring city, C,
        that has not yet been visited."""
        if start is None: 
            start = self.first(cities)
        
        tour = [start]
        unvisited = set(cities - {start})
        while unvisited:
            C = self.nearest_neighbor(tour[-1], unvisited)
            tour.append(C)
            unvisited.remove(C)
        return tour


    def nearest_neighbor(self, A, cities):
        "Find the city in cities that is nearest to city A."
        nearest_distance = float("inf")
        for city in cities:
            if self.distance(A,city) < nearest_distance:
                nearest_distance = self.distance(A,city)
                nearest_neighbor = city 
        return nearest_neighbor
    
    
    def repeated_nn_tsp(self,cities, repetitions=None):
        "Repeat the nn_tsp algorithm starting from each city; return the shortest tour."
        return self.shortest_tour(self.nn_tsp(self.cities, start) 
                                  for start in self.sample(self.cities, repetitions))
    
    def sample(self, population, k, seed=42):
        "Return a list of k elements sampled from population. Set random.seed with seed."
        if k is None or k > len(population): 
            return population
        random.seed(len(population) * k * seed)
        return random.sample(population, k)
    
    def repeat_rep_nn_tsp(self,cities, repetitions): 
        return self.repeated_nn_tsp(self.cities, repetitions)                              
                                  
    
    def greedy_tsp(self,cities):
        """Go through edges, shortest first. Use edge to join segments if possible."""
        endpoints = {c: [c] for c in self.cities} 
        for (A, B) in self.shortest_edges_first(self.cities):
            if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
                new_segment = self.join_endpoints(endpoints, A, B)
                if len(new_segment) == len(self.cities):
                    return new_segment
    
    def shortest_edges_first(self, cities):
        "Return all edges between distinct cities, sorted shortest first."
        edges = [(A, B) for A in self.cities for B in self.cities 
                        if id(A) < id(B)]
        return sorted(edges, key=lambda edge: self.distance(*edge))
    
    def join_endpoints(self, endpoints, A, B):
        "Join B's segment onto the end of A's and return the segment. Maintain endpoints dict."
        Asegment, Bsegment = endpoints[A], endpoints[B]
        if Asegment[-1] is not A: Asegment.reverse()
        if Bsegment[0] is not B: Bsegment.reverse()
        Asegment.extend(Bsegment)
        del endpoints[A], endpoints[B] # A and B are no longer endpoints
        endpoints[Asegment[0]] = endpoints[Asegment[-1]] = Asegment
        return Asegment
    
    
    def altered_greedy_tsp(self,cities):
        "Run greedy TSP algorithm, and alter the results by reversing segments."
        return self.alter_tour(self.greedy_tsp(cities))

    def alter_tour(self, tour):
        "Try to alter tour for the better by reversing segments."
        original_length = self.tour_length(tour)
        for (start, end) in self.all_segments(len(tour)):
            self.reverse_segment_if_better(tour, start, end)
        # If we made an improvement, then try again; else stop and return tour.
        if self.tour_length(tour) < original_length:
            return self.alter_tour(tour)
        return tour

    def all_segments(self, N):
        "Return (start, end) pairs of indexes that form segments of tour of length N."
        return [(start, start + length)
                for length in range(N, 2-1, -1)
                for start in range(N - length + 1)]
    
    def reverse_segment_if_better(self, tour, i, j):
        "If reversing tour[i:j] would make the tour shorter, then do it." 
        # Given tour [...A-B...C-D...], consider reversing B...C to get [...A-C...B-D...]
        A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
        # Are old edges (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
        if self.distance(A, B) + self.distance(C, D) > self.distance(A, C) + self.distance(B, D):
            tour[i:j] = reversed(tour[i:j])
    
    
    def mst_tsp(self, cities):
        "Create a minimum spanning tree and walk it in pre-order, omitting duplicates."
        return self.preorder_traversal(self.mst(cities), self.first(cities))

    def mst(self, vertexes):
        """Given a set of vertexes, build a minimum spanning tree: a dict of the form {parent: [child...]}, 
        where parent and children are vertexes, and the root of the tree is first(vertexes)."""
        tree  = {self.first(vertexes): []} # the first city is the root of the tree.
        edges = self.shortest_edges_first(vertexes)
        while len(tree) < len(vertexes):
            (A, B) = self.shortest_usable_edge(edges, tree)
            tree[A].append(B)
            tree[B] = []
        return tree

    def shortest_usable_edge(self, edges, tree):
        "Find the ehortest edge (A, B) where A is in tree and B is not."
        (A, B) = self.first((A, B) for (A, B) in edges if (A in tree) ^ (B in tree)) # ^ is "xor" 
        return (A, B) if (A in tree) else (B, A)
    
    def preorder_traversal(self, tree, root):
        "Traverse tree in pre-order, starting at root of tree."
        result = [root]
        for child in tree.get(root, ()):
            result.extend(self.preorder_traversal(tree, child))
        return result
    
    def alter_tour(self,tour):
        "Try to alter tour for the better by reversing segments."
        original_length = self.tour_length(tour)
        for (start, end) in self.all_segments(len(tour)):
            self.reverse_segment_if_better(tour, start, end)
        # If we made an improvement, then try again; else stop and return tour.
        if self.tour_length(tour) < original_length:
            return self.alter_tour(tour)
        return tour
    
    def altered_mst_tsp(self,cities): 
        return self.alter_tour(self.mst_tsp(cities))







def main():
    "Read imported dataset with the coordinates"
    df = read_csv("Data/gr9882.tsp", sep=" ",skiprows=8, skipfooter=1,engine='python', usecols = [1,2],names=['X','Y'])
    
    num_cities = int(input("Please enter the number of cities for optimal solutions:(Any integer up to 11 seema a wise option)"))
    
    "-------------------Optimal solutions----------------"
    ObjOpt = AllMethodsTSP_VLSI(df[:num_cities], "alltours_tsp")
    
    print("Calculating solutions for {} cities. Please wait..".format(num_cities))
    ObjOpt.TSP_solution()
    print("......")
    algorithms = ["hk_tsp", "alltours_tsp"]
    
    print("Comparing the performance of the algorithms based on the imported dataset. Please wait..")
    ObjOpt.benchmarks(algorithms, df[:num_cities])
    print("......")

    "-----------------------------------"
    

    "Read imported dataset containing the coordinates"
    df2 = read_csv("Data/xit1083.tsp", sep=" ",skiprows=8, skipfooter=1,engine='python', usecols = [1,2],names=['X','Y'])


    "-------------------Near-Optimal solutions----------------"
    ObjNearOpt = AllMethodsTSP_VLSI(df2, "altered_mst_tsp")
    ObjNearOpt.TSP_solution()
    print("......")


    ObjNearOpt = AllMethodsTSP_VLSI(df2)

    algos = ["nn_tsp", "greedy_tsp"]

    print("Comparing the performance of the algorithms based on the imported dataset. Please wait..")
    ObjNearOpt.benchmarks(algos, df2)
    "-----------------------------------"



if __name__ == '__main__':
    main()