import csv
import sys
import collections
from random import randint
import numpy as np
import time
import math
#import rtree.index
from scipy import spatial
import pymongo
import mongoConnect
import pprint
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from copy import deepcopy
import evaluation_measures


class DBSCAN:
	def __init__(self, eps, minPts, dataset):

		#int
		self.eps = eps

		#int
		self.minPts = minPts

		#n dimensional numpy array np.array((datasetSize, numberOfDimensions, label = initially 0)))
		#labels: 0 = unclassified, -1 noise, any other number = the cluster number
		self.dataset = dataset

		#dictionary - 0 = unclassified, -1 = noise, other number = the cluster id
		self.clusters = collections.defaultdict(list)

	def doDbscan(self):
		
		clusterId = 1

		for obj in self.dataset:
			if obj[-1] == 0:
				if self.expandCluster(obj, clusterId):
					clusterId = clusterId + 1

	def distEuclid(self, a, b):
		return np.linalg.norm(a-b)

	def regionQuery(self, obj):
		result = []
		for obj2 in dataset:
			if(self.distEuclid(obj, obj2) <= self.eps) :
				result.append(obj2)
		return np.array(result)

	def expandCluster(self, obj, clusterId):

		neighs = self.regionQuery(obj)
		if (len(neighs) < self.minPts) :
			obj[-1] = -1
			return False
		neigh = neighs[0]
		neigh[-1] = clusterId
		neighs = np.delete(neighs, 0, 0) #delete the clustered point: idx = 0, axis = 0
		while (len(neighs) != 0) :
			neigh = neighs[0]
			results = self.regionQuery(neigh)

			if(len(results) >= self.minPts):
				for result in results:
					if (result[-1] == 0 or result[-1] == -1) :
						if (result[-1] == 0) :
							np.append(neighs, [result], 0) #append something with the same dimensions on axis 0
						result[-1] = clusterId
			neighs = np.delete(neighs, 0, 0)


class DBSCANRtree(DBSCAN):

	def buildIndex(self, boundingBoxX, boundingBoxY):
		self.boundingBoxX = boundingBoxX
		self.boundingBoxY = boundingBoxY

		self.rTreeIndex = rtree.index.Rtree()
		uniqueKey = 0
		for coord in self.dataset:
			uniqueKey = uniqueKey + 1
			left = int(coord[0] - boundingBoxX)
			bottom = int(coord[1] - boundingBoxY)
			right = int(coord[0] + boundingBoxX)
			top = int(coord[1] + boundingBoxY)
			self.rTreeIndex.add(uniqueKey, (left, bottom, right, top), obj={'datasetObject':coord})

	def regionQuery(self, obj):
		result = []
		objToBoundingBox = (int(obj[0] - self.boundingBoxX), int(obj[1] - self.boundingBoxY), int(obj[0] + self.boundingBoxX), int(obj[1] + self.boundingBoxY))
		intermediaryResults = [
		(i.id, i.object) 
		for i in self.rTreeIndex.nearest(objToBoundingBox, self.eps, objects=True)
		]
		for intermediaryResult in intermediaryResults:
			result.append(intermediaryResult[1]['datasetObject'])

		return np.array(result)

class DBSCANKDtree(DBSCAN):

	def buildIndex(self):
		datasetForIndex = np.delete(self.dataset, 2, 1) #eliminate the third dimension which is the cluster id
		self.kdIndex = spatial.KDTree(datasetForIndex, leafsize=1000)

	def regionQuery(self, obj):
		result = []
		neighIds = self.kdIndex.query_ball_point([obj[0], obj[1]], self.eps)
		for neighId in neighIds:
			result.append(self.dataset[neighId])
		return np.array(result)

class DBSCANKDtreeSimilarityJoin:
	def __init__(self, eps, dataset):
		self.eps = eps
		self.dataset = dataset
		#clean collection

		#number of dimension 
		self.no_dims = 2
		
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB")
		self.mongoConnectInstance.dropCollection("kdTreeDBSCAN")

	def cleanup(self):
		self.mongoConnectInstance.dropCollection("kdTreeDBSCAN")

	def buildIndex(self):
		datasetForIndex = np.delete(self.dataset, 2, 1) #eliminate the third dimension which is the cluster id
		self.kdIndex = spatial.KDTree(datasetForIndex, leafsize=1000)

	def createEpsChains(self):
		results = self.kdIndex.query_ball_tree(self.kdIndex, self.eps)
		for idx in range(len(self.dataset)):
			for result in self.dataset[results[idx]]:
				self.upsertPixelValue('kdTreeDBSCAN', {"$or":[ {"bucket":[]},{"bucket": [result[0], result[1]] }] }, [[result[0], result[1]], [self.dataset[idx][0], self.dataset[idx][1]]], False)
				self.upsertPixelValue('kdTreeDBSCAN', {"$or":[ {"bucket":[]},{"bucket": [self.dataset[idx][0], self.dataset[idx][1]] }] }, [[result[0], result[1]], [self.dataset[idx][0], self.dataset[idx][1]]])
	
	def finalFindAndMerge(self):
		for obj in self.dataset:
			self.findAndMerge('kdTreeDBSCAN', obj)

	def plotClusters(self):
		cursor = self.mongoConnectInstance.getRecords("kdTreeDBSCAN", {}, {"bucket"})
		for document in cursor:
			print(document)
			coordsInDocument = list()
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				plt.scatter(pair[0], pair[1], c=color)
				#plt.text(pair[0], pair[1], str(pair[0])+', '+str(pair[1]))
		plt.show()

	def evaluateClusters(self, clase_points):

		evaluation_dict = {}
		point2cluster = {}
		point2class = {}

		cluster_points = collections.defaultdict(list)
		cursor = self.mongoConnectInstance.getRecords("kdTreeDBSCAN", {}, {"bucket"})
		clusterId = 1
		for document in cursor:
			coordsInDocument = list()
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				cluster_points[clusterId].append( (pair[0], pair[1], clusterId) )
			clusterId = clusterId + 1

		idx = 0
		for elem in clase_points:
			evaluation_dict[idx] = {}
			for points in clase_points[elem]:
				point2class[points] = idx
			idx += 1

		idx = 0
		for elem in cluster_points:
			for point in cluster_points[elem]:
				index_dict = list()
				for dim in range(self.no_dims):
					index_dict.append(point[dim])
				point2cluster[tuple(index_dict)] = idx
			for c in evaluation_dict:
				evaluation_dict[c][idx] = 0
			idx += 1

		for point in point2cluster:
			evaluation_dict[point2class[point]][point2cluster[point]] += 1
				

		print('Purity:  ', evaluation_measures.purity(evaluation_dict))
		print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
		print('RI       ', evaluation_measures.rand_index(evaluation_dict))
		print('ARI      ', evaluation_measures.adj_rand_index(evaluation_dict))

		f = open("rezultate_evaluare.txt", "a")
		f.write('kdTreeDBSCAN'+"\n");
		f.write('Purity:  '+str(evaluation_measures.purity(evaluation_dict))+"\n")
		f.write('Entropy:  '+str(evaluation_measures.entropy(evaluation_dict))+"\n")
		f.write('RI:  '+str(evaluation_measures.rand_index(evaluation_dict))+"\n")
		f.write('ARI:  '+str(evaluation_measures.adj_rand_index(evaluation_dict))+"\n")
		f.close()
	
	def upsertPixelValue(self, collection, filter, epsNeigh, upsert = True):
		self.mongoConnectInstance.update(collection, filter, {"$addToSet":{"bucket":{"$each":epsNeigh}}}, upsert, True)

	def findAndMerge(self, collection, coord):
		#aggregate the results
		if(self.mongoConnectInstance.count(collection, {"bucket": [coord[0], coord[1]] } ) <= 1):
			return
		aggregationString=[{"$match": {"bucket": [coord[0], coord[1]] } },{"$unwind": "$bucket"},{"$group" : {"_id" : ObjectId(), "bucket":{"$addToSet":"$bucket"}}}]
		aggregationResult = self.mongoConnectInstance.aggregate(collection, aggregationString)
		aggregationResultList = list(aggregationResult)

		#remove all other documents - we aggregated them
		self.mongoConnectInstance.remove(collection, {"bucket": [coord[0], coord[1]] })
		#insert the aggregated document
		for document in aggregationResultList:
			self.mongoConnectInstance.insert(collection, document)

	def doDbscan(self):
		self.createEpsChains()
		self.finalFindAndMerge()

class quickDBSCAN:
	def __init__(self, eps, pivotChoosingStrategy):
		#int
		self.eps = eps

		#number of dimension 
		self.no_dims = 2

		#pivot choosing strategy: 1 = corner, 2 = random
		self.pivotChoosingStrategy = pivotChoosingStrategy

		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB")
		#clean collection
		self.mongoConnectInstance.dropCollection("quickDBSCAN")

	
	def cleanup(self):
		self.mongoConnectInstance.dropCollection("quickDBSCAN")

	def randomObject(self, objs):
		randomIndex = randint(0, len(objs)-1)
		return objs[randomIndex]

	def euclideanDistPositionNumpy(self, a, b):
		return np.linalg.norm(a-b)

	def euclideanDistPosition(self, a, b):
		return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

	def areEqual(self, a, b):
		if (a[0] == b[0] and a[1] == b[1]):
			return True
		else:
			return False

	def swapper(self, a, b):
		return (copy(b), copy(a))

	def ball_average(self, objs, p1):
		avgDistHelper = []
		for coord1 in objs:
			for coord2 in objs:
				if( coord1 != coord2 ): #pixel != p1 in numpy arrays
					avgDistHelper.append(self.euclideanDistPosition(coord1, coord2))
		avgDistHelper = np.array(avgDistHelper)
		return sum(avgDistHelper)/len(avgDistHelper)

	def ball_median(self, objs, p1):
		#print("len(objs) "+str(len(objs)))
		avgDistHelper = []
		for coord1 in objs:
			if( coord1 != p1 ): #pixel != p1 in numpy arrays
				avgDistHelper.append(self.euclideanDistPosition(coord1, p1))
		avgDistHelper = np.array(avgDistHelper)
		return np.median(avgDistHelper)

	def centeroidnp(self, arr):
		arr = np.array(arr)
		length = arr.shape[0]
		sum_x = np.sum(arr[:, 0])
		sum_y = np.sum(arr[:, 1])
		return [round(sum_x/length, 2), round(sum_y/length, 2)]

	def medoidnp(self, objs):
		distMatrix = np.zeros((len(objs), len(objs)))
		for i in range(len(objs)):
			for j in range(len(objs)):
				distMatrix[i, j] = self.euclideanDistPosition(objs[i], objs[j])
		distMatrix = np.array(distMatrix)
		return objs[np.argmin(distMatrix.sum(axis=0))]

	def furthestPivot(self, objs):
		helper = self.randomObject(objs)
		maxDist = 0
		pivot = None
		for obj in objs:
			dist = self.euclideanDistPosition(obj, helper)
			if(dist > maxDist):
				maxDist = dist
				pivot = obj
		return pivot

	def farthestObjectPivotDistance(self, objs, p1):
		maxDist = 0
		for obj in objs:
			dist = self.euclideanDistPosition(obj, p1)
			if(dist > maxDist):
				maxDist = dist
		return maxDist/2

	def bounding_box_naive(self, points):
		"""returns a list containing the bottom left and the top right 
		points in the sequence
		Here, we use min and max four times over the collection of points
		"""
		bot_left_x = min(point[0] for point in points)
		bot_left_y = min(point[1] for point in points)
		top_right_x = max(point[0] for point in points)
		top_right_y = max(point[1] for point in points)

		return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]

	def cornerVantagePoint(self, dataset):
		corners = self.bounding_box_naive(dataset)
		return [corners[0][0], corners[0][1]]

	def partition(self, objs, p1):
		partL = []
		partG = []
		winL = []
		winG = []
		
		r = round(self.ball_median(objs, p1), 2)
	
		startIdx = 0
		endIdx = len(objs)-1
		startDist = self.euclideanDistPosition(objs[startIdx], p1)
		endDist = self.euclideanDistPosition(objs[endIdx], p1)

		while(startIdx < endIdx):
		
			while(endDist > r and startIdx < endIdx):
				if(endDist <= r+self.eps):
					helper1 = deepcopy(objs[endIdx])
					winG.append(helper1)
				endIdx = endIdx - 1
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
				
			while(startDist <= r and startIdx < endIdx):
				if(startDist >= r-self.eps):
					helper2 = deepcopy(objs[startIdx])
					winL.append(helper2)
				startIdx = startIdx + 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				
			if(startIdx < endIdx):
				if(endDist >= r-self.eps):
					helper3 = deepcopy(objs[endIdx])
					winL.append(helper3)
				if(startDist <= r+self.eps):
					helper4 = deepcopy(objs[startIdx])
					winG.append(helper4)
				#exchange items
				objs[startIdx], objs[endIdx] = objs[endIdx], objs[startIdx]
				startIdx = startIdx + 1
				endIdx = endIdx - 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
		if(startIdx == endIdx):
			if(endDist > r and endDist <= r+self.eps):
				helper5 = deepcopy(objs[endIdx])
				winG.append(helper5)
			if(startDist <= r and startDist >= r-self.eps):
				helper6 = deepcopy(objs[startIdx])
				winL.append(helper6)

			if(endDist > r):
				endIdx = endIdx - 1

		#create partL and partG relative to the distance from p1
		for obj in objs:
			if(self.euclideanDistPosition(obj, p1) < r):
				partL.append(obj)
			else:
				partG.append(obj)
		return (partL, partG, winL, winG)

	def quickJoin(self, objs, constSmallNumber):
		objs = list(set(objs))

		if(len(objs) == 0):
			return
		if(len(objs) < constSmallNumber):
			self.nestedLoop(objs)
			return

		if (self.pivotChoosingStrategy == 1):
			p1 = self.cornerVantagePoint(objs)
		else:
			p1 = self.randomObject(objs)
		
		(partL, partG, winL, winG) = self.partition(objs, p1)
		
		self.quickJoinWin(winL, winG, constSmallNumber)

		#if one of the partitions is 0, just stop and do the nested loop on the other one
		if(len(partL) == 0):
			self.nestedLoop(partG)

		if(len(partG) == 0):
			self.nestedLoop(partL)

		if(len(partL) == 0 or len(partG) == 0):
			return

		self.quickJoin(partL, constSmallNumber)
		self.quickJoin(partG, constSmallNumber)

	def quickJoinWin(self, objs1, objs2, constSmallNumber):
		objs1 = list(set(objs1))
		objs2 = list(set(objs2))
	
		if (len(objs1) == 0 or len(objs2) == 0):
			return

		if (len(objs1) == 1 or len(objs2) == 1):
			self.nestedLoop2(objs1, objs2)
			return

		totalLen = len(objs1) + len(objs2)
	
		if(totalLen < constSmallNumber):
			self.nestedLoop2(objs1, objs2, ["objs1", "objs2"])
			return

		allObjects = objs1 + objs2

		if (self.pivotChoosingStrategy == 1):
			p1 = self.cornerVantagePoint(allObjects)
		else:
			p1 = self.randomObject(allObjects)

		(partL1, partG1, winL1, winG1) = self.partition(objs1, p1)
		(partL2, partG2, winL2, winG2) = self.partition(objs2, p1)

		#if any of the pairs contains a zero, switch to brute force
		if (len(partL1) == 0 or len(partL2) == 0 or len(partG1) == 0 or len(partG2) == 0 or len(winL1) == 0 or len(winL2) == 0 or len(winG1) == 0 or len(winG2) == 0):
			self.nestedLoop2(winL1, winG2, ["winL1", "winG2"])
			self.nestedLoop2(winG1, winL2, ["winG1", "winL2"])
			self.nestedLoop2(partL1, partL2, ["partL1", "winL2"])
			self.nestedLoop2(partG1, partG2, ["partG11", "winG2"])
			return

		self.quickJoinWin(winL1, winG2, constSmallNumber)
		self.quickJoinWin(winG1, winL2, constSmallNumber)
		self.quickJoinWin(partL1, partL2, constSmallNumber)
		self.quickJoinWin(partG1, partG2, constSmallNumber)

	def nestedLoop(self, objs):
		for coord1 in objs:
			for coord2 in objs:
				if( self.euclideanDistPosition(coord1, coord2) <= self.eps and  coord1 != coord2):
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord1[0], coord1[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord2[0], coord2[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					
	def nestedLoop2(self, objs1, objs2, deUndeVine = []):
		for coord1 in objs1:
			for coord2 in objs2:
				if( self.euclideanDistPosition(coord1, coord2) <= self.eps and coord1 != coord2):
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord1[0], coord1[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord2[0], coord2[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					
	def upsertPixelValue(self, collection, filter, epsNeigh):
		self.mongoConnectInstance.update(collection, filter, {"$addToSet":{"bucket":{"$each":epsNeigh}}}, True, True)

	def findAndMerge(self, collection, coord):
		#aggregate the results
		if(self.mongoConnectInstance.count(collection, {"bucket": [coord[0], coord[1]] } ) <= 1):
			return
		aggregationString=[{"$match": {"bucket": [coord[0], coord[1]] } },{"$unwind": "$bucket"},{"$group" : {"_id" : ObjectId(), "bucket":{"$addToSet":"$bucket"}}}]
		aggregationResult = self.mongoConnectInstance.aggregate(collection, aggregationString)
		aggregationResultList = list(aggregationResult)
		
		self.mongoConnectInstance.remove(collection, {"bucket": [coord[0], coord[1]] })

		#insert the aggregated document
		for document in aggregationResultList:
			self.mongoConnectInstance.insert(collection, document)

	def finalFindAndMerge(self, objs):
		for obj in objs:
			self.findAndMerge("quickDBSCAN", obj)

	def plotClusters(self):
		cursor = self.mongoConnectInstance.getRecords("quickDBSCAN", {}, {"bucket"})
		for document in cursor:
			coordsInDocument = list()
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				plt.scatter(pair[0], pair[1], c=color)
				#plt.text(pair[0], pair[1], str(pair[0])+', '+str(pair[1]))
				
		plt.show()

	def evaluateClusters(self, clase_points):

		evaluation_dict = {}
		point2cluster = {}
		point2class = {}

		cluster_points = collections.defaultdict(list)
		cursor = self.mongoConnectInstance.getRecords("quickDBSCAN", {}, {"bucket"})
		clusterId = 1
		for document in cursor:
			coordsInDocument = list()
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				cluster_points[clusterId].append( (pair[0], pair[1], clusterId) )
			clusterId = clusterId + 1

		idx = 0
		for elem in clase_points:
			evaluation_dict[idx] = {}
			for points in clase_points[elem]:
				point2class[points] = idx
			idx += 1

		idx = 0
		for elem in cluster_points:
			for point in cluster_points[elem]:
				index_dict = list()
				for dim in range(self.no_dims):
					index_dict.append(point[dim])
				point2cluster[tuple(index_dict)] = idx
			for c in evaluation_dict:
				evaluation_dict[c][idx] = 0
			idx += 1

		for point in point2cluster:
			evaluation_dict[point2class[point]][point2cluster[point]] += 1
				

		print('Purity:  ', evaluation_measures.purity(evaluation_dict))
		print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
		print('RI       ', evaluation_measures.rand_index(evaluation_dict))
		print('ARI      ', evaluation_measures.adj_rand_index(evaluation_dict))

		f = open("rezultate_evaluare.txt", "a")
		f.write('quickDBSCAN'+"\n");
		f.write('Purity:  '+str(evaluation_measures.purity(evaluation_dict))+"\n")
		f.write('Entropy:  '+str(evaluation_measures.entropy(evaluation_dict))+"\n")
		f.write('RI:  '+str(evaluation_measures.rand_index(evaluation_dict))+"\n")
		f.write('ARI:  '+str(evaluation_measures.adj_rand_index(evaluation_dict))+"\n")
		f.close()



	def returnPairs(self):
		finalPairs = []
		cursor = self.mongoConnectInstance.getRecords("quickDBSCAN", {}, {"bucket"})
		for document in cursor:
			coordsInDocument = list()
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				finalPairs.append((pair[0], pair[1], color))
		return finalPairs;

def createDataset(datasetFilename):
	dataset = list()
	datasetQuick = list()

	with open(datasetFilename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')
		for row in csvReader:
			dataset.append( ( round(float(row[0]), 3), round(float(row[1]), 3), 0) )
			datasetQuick.append( ( round(float(row[0]), 3), round(float(row[1]), 3) ) )

	dataset = np.array( list( set (dataset) ))
	datasetQuick = list( (set (datasetQuick)))

	return (dataset, datasetQuick)

def createDatasetEvaluation(datasetFilename):
	dataset = list()
	datasetQuick = list()
	clase_points = collections.defaultdict(list)

	with open(datasetFilename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')
		for row in csvReader:
			dataset.append( ( round(float(row[0]), 3), round(float(row[1]), 3), 0) )
			datasetQuick.append( ( round(float(row[0]), 3), round(float(row[1]), 3), int(row[2])) )

	dataset = np.array( list( set (dataset) ))
	datasetQuick = list( (set (datasetQuick)))

	for datasetItem in datasetQuick:
		clase_points[datasetItem[2]].append( (datasetItem[0], datasetItem[1]) )

	#strip the class from the dataset
	datasetQuick = [ (element[0], element[1]) for element in datasetQuick]

	return (dataset, datasetQuick, clase_points)

if __name__ == '__main__':

	'''
	noisy_circles_300 = 0.25
	blobls_300 = 1
	noisy_moons_300 = 0.25
	'''

	sys.setrecursionlimit(15000)

def computeRuntimes():

	datasetFiles = ["dataset/noisy_circles_300.csv", "dataset/noisy_moons_300.csv", "dataset/blobs_600.csv", "dataset/noisy_circles_600.csv", "dataset/noisy_moons_600.csv", "dataset/blobs_600.csv", "dataset/noisy_circles_1000.csv", "dataset/noisy_moons_1000.csv", "dataset/blobs_1000.csv"]
	epsValues = [0.1, 0.25, 0.5, 0.8, 1]

	for datasetFile in datasetFiles:
		simpleDBSCANTimes = list()
		kdTreeIdxCreationTimes = list()
		kdTreeDBSCANTimes = list()
		quickDBSCANTimes = list()
		f = open("quickDBSCANPerformance.txt","a+")
		f.write('FILE '+str(datasetFile)+'=====================\n')
		f.write('\n\n')
		for eps in epsValues:
			(dataset, datasetQuick) = createDataset(datasetFile)

			simpleDBSCAN = DBSCAN(eps, 5, dataset)
			start = time.time()
			simpleDBSCAN.doDbscan()
			end = time.time()
			simpleDBSCANTimes.append((end - start))

			dbscan = DBSCANKDtreeSimilarityJoin(eps, dataset)
			dbscan.cleanup()
			start = time.time()
			dbscan.buildIndex()
			end = time.time()
			kdTreeIdxCreationTimes.append((end - start))

			start = time.time()
			dbscan.doDbscan()
			end = time.time()
			kdTreeDBSCANTimes.append((end - start))

			quickDBSCANInstance = quickDBSCAN(eps, 1)
			quickDBSCANInstance.cleanup()
			start = time.time()
			quickDBSCANInstance.quickJoin(datasetQuick, 10)
			quickDBSCANInstance.finalFindAndMerge(datasetQuick)

			end = time.time()
			quickDBSCANTimes.append((end - start))

		f.write('\n')
		f.write('simpleDBSCANTimes times for all eps '+str(simpleDBSCANTimes)+'\n')

		f.write('Index creation times for all eps '+str(kdTreeIdxCreationTimes)+'\n')
		f.write('\n')

		f.write('kdTree times for all eps '+str(kdTreeIdxCreationTimes)+'\n')
		f.write('\n')

		f.write('quickDBSCAN times for all eps '+str(quickDBSCANTimes)+'\n')
		
		f.close()

def computeEvaluation():

	datasetFiles = [("dataset/noisy_circles_300_evaluation.csv", 0.25), ("dataset/noisy_moons_300_evaluation.csv", 0.25), ("dataset/blobs_600_evaluation.csv", 1), ("dataset/noisy_circles_600_evaluation.csv", 0.25), ("dataset/noisy_moons_600_evaluation.csv", 0.25), ("dataset/blobs_600_evaluation.csv", 1), ("dataset/noisy_circles_1000_evaluation.csv", 0.25), ("dataset/noisy_moons_1000_evaluation.csv", 0.25), ("dataset/blobs_1000_evaluation.csv", 1)]

	for datasetItem in datasetFiles:
		datasetFile = datasetItem[0]
		eps = datasetItem[1]

		(dataset, datasetQuick, clase_points) = createDatasetEvaluation(datasetFile)

		dbscan = DBSCANKDtreeSimilarityJoin(eps, dataset)
		dbscan.cleanup()
		dbscan.buildIndex()
		dbscan.doDbscan()
		dbscan.evaluateClusters(clase_points)
		
		quickDBSCANInstance = quickDBSCAN(eps, 1)
		quickDBSCANInstance.cleanup()
		quickDBSCANInstance.quickJoin(datasetQuick, 10)
		quickDBSCANInstance.finalFindAndMerge(datasetQuick)
		quickDBSCANInstance.evaluateClusters(clase_points)

def plotFiles():

	datasetFilesPlot = [ ("dataset/noisy_circles_300.csv", 0.25), ("dataset/noisy_moons_300.csv", 0.25), ("dataset/blobs_300.csv", 1)]
	subplotNr = 1;
	figId = ['a', 'b', 'c']
	for datasetFileEps in datasetFilesPlot:
		datasetFile = datasetFileEps[0]
		eps = datasetFileEps[1]
		(dataset, datasetQuick) = createDataset(datasetFile)

		quickDBSCANInstance = quickDBSCAN(eps, 1)
		quickDBSCANInstance.cleanup()
		quickDBSCANInstance.quickJoin(datasetQuick, 10)
		quickDBSCANInstance.finalFindAndMerge(datasetQuick)
		finalPairs = quickDBSCANInstance.returnPairs()

		plt.subplot(1, 3, subplotNr) #number of rows, number of columns, and which subplot you're currently on
		
		for finalPair in finalPairs:
			plt.scatter(finalPair[0], finalPair[1], c=finalPair[2])
		plt.title('Fig 1 ('+str(figId[subplotNr-1])+')')

		subplotNr = subplotNr + 1;

	plt.tight_layout()
	plt.show()

def pivotDifferences():

	datasetFilesPlot = [ ("dataset/noisy_circles_300.csv", 0.25), ("dataset/noisy_moons_300.csv", 0.25), ("dataset/blobs_300.csv", 1)]
	subplotNr = 1;
	figId = ['a', 'b', 'c']
	for datasetFileEps in datasetFilesPlot:
		datasetFile = datasetFileEps[0]
		eps = datasetFileEps[1]
		(dataset, datasetQuick) = createDataset(datasetFile)
		
		quickDBSCANInstance = quickDBSCAN(eps, 1)
		quickDBSCANInstance.cleanup()
		start = time.time()
		quickDBSCANInstance.quickJoin(datasetQuick, 10)
		quickDBSCANInstance.finalFindAndMerge(datasetQuick)
		end = time.time()
		print('Vantage, File '+ str(datasetFile) + ' Time ' + str(end-start));

		quickDBSCANInstance = quickDBSCAN(eps, 2)
		quickDBSCANInstance.cleanup()
		start = time.time()
		quickDBSCANInstance.quickJoin(datasetQuick, 10)
		quickDBSCANInstance.finalFindAndMerge(datasetQuick)
		end = time.time()
		print('Random, File '+ str(datasetFile) + ' Time ' + str(end-start));

computeEvaluation()






			
			





