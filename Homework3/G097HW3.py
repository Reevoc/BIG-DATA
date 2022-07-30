# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------
    time_start = time.time()
    coreset = points.mapPartitions(lambda x: extractCoreset(x, k ,z ))
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    elems = coreset.collect()
    time_r1 = time.time() - time_start
    time_start = time.time()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # ****** ADD YOUR CODE
    solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, alpha=2)
    time_r2 = time.time() - time_start
    print("Time Round 1: ", time_r1 * 1000, " ms")
    print("Time Round 2: ", time_r2 * 1000, " ms")
    return solution

    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
     
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, k , z):
    partition = list(iter)
    if len(partition) == 0 :
        return []
    centers = kCenterFFT(partition, k+z+1)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&weigcoresetWeightshts&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(points, weights, k, z, alpha = 2):
    iteration = 0
    r_min = compute_rmin(points.copy()[:z+k+1])
    r=r_min
    solution = []
    while True:
        iteration += 1
        solution = []
        tmp_points = points.copy()
        tmp_weights = weights.copy()
        free_points_weight = sum(tmp_weights)
        # free_points is now a list of touples that contain both weight and position, so non more two lists
        free_points = list(zip(tmp_points, tmp_weights))
        inside_iter = 0
        while (len(solution) < k) and (free_points_weight > 0):
            inside_iter += 1
            maxim = -1
            new_center = None
            for point, weight in free_points:
                ball_weight = compute_ball_weight(
                    point, free_points, r, alpha)
                if ball_weight > maxim:
                    maxim = ball_weight
                    new_center = point 
                    new_center_weight = weight
            solution.append(new_center)
            free_points.remove((new_center, new_center_weight))
            free_points_weight -= new_center_weight
            new_points = free_points.copy()
            for point, weight in new_points:
                distance = euclidean(new_center, point)
                if distance < (3+(4*alpha))*r:
                    free_points.remove((point, weight))
                    free_points_weight -= weight
        if free_points_weight <= z:
            print("Initial guess = ", r_min)
            print("Final guess = ", r)
            print("Number of guesses = ", iteration)
            return solution
        else:
            r = 2*r

def compute_ball_weight(center, free_points, r, alpha):
    #inefficent, precomputer distances are probably better
    ball_weight = 0
    for point, weight in free_points:
        distance = euclidean(center, point)
        if distance <= (1+(2*alpha))*r:
            ball_weight += weight
    return ball_weight

def compute_rmin(points):
    #jesus christ
    rmin = math.inf
    for point1 in points:
        for point2 in points:
            if point1 != point2:
                dist=euclidean(point1, point2)
                if dist < rmin:
                    rmin = dist
    return rmin/2

#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(inputPoints, solution, z):
    distances = []
    count = 0
    inputPoints = inputPoints.collect()
    for point in inputPoints:
        minimum = math.inf
        for cluster in solution: 
                dist = euclidean(cluster, point)
                minimum = min(minimum, dist) 
        #if less distance of z+1    
        if len(distances) < z + 1 and count == 0:
            distances.append(minimum)
            if len(distances) == z+1 and count == 0:
                distances = sorted(distances, reverse=False)
                count = 1   
        else:
            for i in range(0, len(distances)):
                if (minimum > distances[i]):
                    distances.insert(i, minimum)
                    distances.pop()
                    break
    return distances[-1]
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#


# Just start the main program
if __name__ == "__main__":
    main()

