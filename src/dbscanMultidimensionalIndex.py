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

class quickDBSCAN:
	def __init__(self, eps):
		#int
		self.eps = eps
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");

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

	def ball_average(self, objs, p1):
		#print("len(objs) "+str(len(objs)))
		avgDistHelper = []
		for coord1 in objs:
			if( areEqual(coord1, p1) == False ): #pixel != p1 in numpy arrays
				avgDistHelper.append(self.euclideanDistPosition(coord1, p1))
		avgDistHelper = np.array(avgDistHelper)
		return sum(avgDistHelper)/len(avgDistHelper)

	def ball_median(self, objs, p1):
		#print("len(objs) "+str(len(objs)))
		avgDistHelper = []
		for coord1 in objs:
			if( (coord1-p1!= 0).any() ): #pixel != p1 in numpy arrays
				avgDistHelper.append(self.euclideanDistPosition(coord1, p1))
		avgDistHelper = np.array(avgDistHelper)
		return np.median(avgDistHelper)

	def centeroidnp(self, arr):
		arr = np.array(arr)
		length = arr.shape[0]
		sum_x = np.sum(arr[:, 0])
		sum_y = np.sum(arr[:, 1])
		return [sum_x/length, sum_y/length]

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

	def partition(self, objs, p1, pivotTrialCount = 0):
		#print("PARTITION len(objs) "+str(len(objs)))
		partL = [] 
		partG = []
		winL = []
		winG = []
		#sorted as reffering to p1
		#objs = np.array(sorted(objs, key=lambda x: math.sqrt((x[0] - 0) ** 2 + (x[1] - 0) ** 2)))

		print("p1 = "+str(p1)+" len objs = "+ str(len(objs)))

		r = self.farthestObjectPivotDistance(objs, p1)
		print("r is "+str(r))
		startIdx = 0
		endIdx = len(objs)-1
		startDist = self.euclideanDistPosition(objs[startIdx], p1)
		endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
		while(startIdx < endIdx):
			while(endDist > r and startIdx < endIdx):
				if(endDist <= r+self.eps):
					winG.append(objs[endIdx])
				endIdx = endIdx - 1
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
				
			while(startDist <= r and startIdx < endIdx):
				if(startDist >= r-self.eps):
					winL.append(objs[startIdx])
				startIdx = startIdx + 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				
			if(startIdx < endIdx):
				if(endDist >= r-self.eps):
					winL.append(objs[endIdx])
				if(startDist <= r+self.eps):
					winG.append(objs[startIdx])
				#exchange items
				objs[startIdx], objs[endIdx] = objs[endIdx], objs[startIdx]
				startIdx = startIdx + 1
				endIdx = endIdx - 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
		if(startIdx == endIdx):
			if(endDist > r and endDist <= r+self.eps):
				winG.append(objs[endIdx])
			if(startDist <= r and startDist >= r-self.eps):
				winL.append(objs[startIdx])
			if(endDist > r):
				endIdx = endIdx - 1

		winL = np.array(winL)
		winG = np.array(winG)

		print("===================================OBJS after")
		print(str(objs))
		print("===================================OBJS end")

		#if I tried to find a good partitioning - no zeros, but I reached the 10th trial, stop
		if(pivotTrialCount >= 10):
			return (objs[0:endIdx], objs[endIdx:len(objs)], winL, winG)

		#if one part of the array is 0, we need a different partitioning to avoid loops
		if(len(objs[0:endIdx])==0 or len(objs[endIdx:len(objs)])==0):
			p1 = self.randomObject(objs)
			pivotTrialCount = pivotTrialCount + 1
			self.partition(objs, p1, pivotTrialCount)

		return (objs[0:endIdx], objs[endIdx:len(objs)], winL, winG)

	def quickJoin(self, objs, constSmallNumber):
		print("quick len(objs), constSmallNumber "+str(len(objs))+" "+str(constSmallNumber))
		if(len(objs) == 0):
			return
		if(len(objs) < constSmallNumber):
			#print("GATA! len(objs) "+str(len(objs)))
			self.nestedLoop(objs)
			return

		p1 = self.randomObject(objs)
		#p1 = objs.max(axis=0)
		
		(partL, partG, winL, winG) = self.partition(objs, p1)
		
		#if(len(winL)>0 and len(winG)>0):
		self.quickJoinWin(winL, winG, constSmallNumber)

		#if one of the partitions is 0, just stop and do the nested loop on the other one
		if(len(partL) == 0):
			self.nestedLoop(partG)
			return

		if(len(partG) == 0):
			self.nestedLoop(partL)
			return

		self.quickJoin(partL, constSmallNumber)
		self.quickJoin(partG, constSmallNumber)

	def quickJoinWin(self, objs1, objs2, constSmallNumber):
		print("Intra in win")
		totalLen = len(objs1) + len(objs2)
		print("win len(objs), constSmallNumber "+str(totalLen)+" "+str(constSmallNumber))
		if(totalLen < constSmallNumber):
			#print("GATA Win! len(objs) "+str(totalLen))
			self.nestedLoop2(objs1, objs2)
			return

		if(len(objs1) <= 1):
			if(len(objs1) == 1):
				self.nestedLoop2(objs1, objs2)
			self.nestedLoop(objs2)
			return

		if(len(objs2) <= 1):
			if(len(objs2) == 1):
				self.nestedLoop2(objs1, objs2)
			self.nestedLoop(objs1)
			return	

		#print("win len objs1 " + str(len(objs1)))
		#print("win len objs2 " + str(len(objs2)))

		allObjects = np.concatenate((objs1, objs2), axis=0)

		p1 = self.randomObject(allObjects)
		#p1 = allObjects.max(axis=0)

		(partL1, partG1, winL1, winG1) = self.partition(objs1, p1)
		(partL2, partG2, winL2, winG2) = self.partition(objs2, p1)

		self.quickJoinWin(winL1, winG2, constSmallNumber)
		self.quickJoinWin(winG1, winL2, constSmallNumber)
		self.quickJoinWin(partL1, partL2, constSmallNumber)
		self.quickJoinWin(partG1, partG2, constSmallNumber)

	def nestedLoop(self, objs):
		for coord1 in objs:
			for coord2 in objs:
				if( self.euclideanDistPosition(coord1, coord2) <= self.eps and  self.euclideanDistPosition(coord1, coord2) != 0):
					self.upsertPixelValue("quickDBSCAN",{"bucket":{"$in":[[],[coord1[0], coord1[1]]]}}, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.upsertPixelValue("quickDBSCAN",{"bucket":{"$in":[[], [coord2[0], coord2[1]]]}}, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					#self.findAndMerge("quickDBSCAN", coord2)
					#self.findAndMerge("quickDBSCAN", coord1)


	def nestedLoop2(self, objs1, objs2):
		for coord1 in objs1:
			for coord2 in objs2:
				if( self.euclideanDistPosition(coord1, coord2) <= self.eps and self.euclideanDistPosition(coord1, coord2) != 0):
					self.upsertPixelValue("quickDBSCAN",{"bucket":{"$in":[[], [coord1[0], coord1[1]]]}}, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.upsertPixelValue("quickDBSCAN",{"bucket":{"$in":[[], [coord2[0], coord2[1]]]}}, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					#self.findAndMerge("quickDBSCAN", coord2)
					#self.findAndMerge("quickDBSCAN", coord1)					


	def upsertPixelValue(self, collection, filter, epsNeigh):
		self.mongoConnectInstance.update(collection, filter, {"$addToSet":{"bucket":{"$each":epsNeigh}}}, True)

	def findAndMerge(self, collection, coord):
		#aggregate the results
		if(self.mongoConnectInstance.count(collection, {"bucket":{"$in":[[coord[0], coord[1]]]}}) <= 1):
			return
		aggregationString=[{"$match":{"bucket":{"$in":[[coord[0], coord[1]]]}}},{"$unwind": "$bucket"},{"$group" : {"_id" : ObjectId(), "bucket":{"$addToSet":"$bucket"}}}]
		aggregationResult = self.mongoConnectInstance.aggregate(collection, aggregationString)
		aggregationResultList = list(aggregationResult)
		
		print("Aggregation")
		#remove all other documents - we aggregated them
		print("Count before remove: "+str(self.mongoConnectInstance.count(collection, {})))
		self.mongoConnectInstance.remove(collection, {"bucket":{"$in":[[coord[0], coord[1]]]}})
		print("Count after remove: "+str(self.mongoConnectInstance.count(collection, {})))
		#insert the aggregated document
		for document in aggregationResultList:
			print("Document")
			self.mongoConnectInstance.insert(collection, document)

	def finalFindAndMerge(self, objs):
		print("In final findAndMerge "+str(len(objs)))
		for obj in objs:
			self.findAndMerge("quickDBSCAN", obj)

	def plotClusters(self):
		cursor = self.mongoConnectInstance.getRecords("quickDBSCAN", {}, {"bucket"})
		for document in cursor:
			print(document)
			coordsInDocument = list()
			for (x, y) in document["bucket"]:
				coordsInDocument.append((x,y))
			coordsInDocument = set(coordsInDocument)
			color = np.random.rand(3,)
			for (x, y) in coordsInDocument:
				plt.scatter(x, y, c=color)
		plt.show()

if __name__ == '__main__':

	sys.setrecursionlimit(15000)

	#read the csv
	datasetFilename = sys.argv[1]
	dataset = list()
	datasetQuick = list()

	with open(datasetFilename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')
		for row in csvReader:
			dataset.append( (float(row[0]), float(row[1]), 0) )
			datasetQuick.append( (float(row[0]), float(row[1])) )

	dataset = np.array(dataset)
	datasetQuick = np.array(datasetQuick)

	print(np.shape(dataset))

	'''dbscan = DBSCANRtree(1, 5, dataset)

	start = time.time()

	dbscan.buildIndex(5, 5)

	end = time.time()

	print('DBSCANRtree buildIndex took '+str(end - start))

	start = time.time()

	dbscan.doDbscan()

	end = time.time()

	print('DBSCANRtree took '+str(end - start))'''

	'''dbscan = DBSCAN(1, 5, dataset)

	start = time.time()

	dbscan.doDbscan()

	end = time.time()

	print('DBSCAN took '+str(end - start))

	dbscan = DBSCANKDtree(1, 5, dataset)

	start = time.time()

	dbscan.buildIndex()

	end = time.time()

	print('DBSCANKdtree buildIndex took '+str(end - start))

	start = time.time()

	dbscan.doDbscan()

	end = time.time()

	print('DBSCANKdtree took '+str(end - start))'''

	quickDBSCAN = quickDBSCAN(4)
	quickDBSCAN.quickJoin(datasetQuick, 10)
	#quickDBSCAN.finalFindAndMerge(datasetQuick)
	quickDBSCAN.plotClusters()




			
			





