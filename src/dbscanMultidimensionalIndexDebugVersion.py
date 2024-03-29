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

	def buildIndex(self):
		datasetForIndex = np.delete(self.dataset, 2, 1) #eliminate the third dimension which is the cluster id
		self.kdIndex = spatial.KDTree(datasetForIndex, leafsize=1000)

	def createEpsChains(self):
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB")
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
	
	def upsertPixelValue(self, collection, filter, epsNeigh, upsert = True):
		self.mongoConnectInstance.update(collection, filter, {"$addToSet":{"bucket":{"$each":epsNeigh}}}, upsert, True)

	def findAndMerge(self, collection, coord):
		#aggregate the results
		if(self.mongoConnectInstance.count(collection, {"bucket": [coord[0], coord[1]] } ) <= 1):
			return
		aggregationString=[{"$match": {"bucket": [coord[0], coord[1]] } },{"$unwind": "$bucket"},{"$group" : {"_id" : ObjectId(), "bucket":{"$addToSet":"$bucket"}}}]
		aggregationResult = self.mongoConnectInstance.aggregate(collection, aggregationString)
		aggregationResultList = list(aggregationResult)
		
		print("Aggregation")
		#remove all other documents - we aggregated them
		print("Count before remove: "+str(self.mongoConnectInstance.count(collection, {})))
		self.mongoConnectInstance.remove(collection, {"bucket": [coord[0], coord[1]] })
		print("Count after remove: "+str(self.mongoConnectInstance.count(collection, {})))
		#insert the aggregated document
		for document in aggregationResultList:
			print("Document")
			self.mongoConnectInstance.insert(collection, document)

	def doDbscan(self):
		self.createEpsChains()
		self.finalFindAndMerge()

class quickDBSCAN:
	def __init__(self, eps):
		#int
		self.eps = eps
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB")
		self.allPairs = []

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

	def partition(self, objs, p1):
		partL = []
		partG = []
		winL = []
		winG = []
		
		print("p1 = "+str(p1)+" len objs = "+ str(len(objs)))

		r = round(self.ball_average(objs, p1), 2)
		print("r is "+str(r))
		#r must not be smaller than eps, it breaks things
		'''if(r < self.eps):
			r = r + 2*self.eps'''
		print("new r is "+str(r))
		startIdx = 0
		endIdx = len(objs)-1
		startDist = self.euclideanDistPosition(objs[startIdx], p1)
		endDist = self.euclideanDistPosition(objs[endIdx], p1)

		while(startIdx < endIdx):
			print("p1 = "+str(p1))
			print("startIdx = "+str(startIdx)+" obj "+str(objs[startIdx]))
			print("endIdx = "+str(endIdx)+" obj "+str(objs[endIdx]))
			print("startDIST = "+str(startDist)+" obj "+str(objs[startIdx]))
			print("endDIST = "+str(endDist)+" obj "+str(objs[endIdx]))

			while(endDist > r and startIdx < endIdx):
				print("PRIMUL WHILE endIdx = "+str(endIdx)+" obj "+str(objs[endIdx])+" endDist "+str(endDist))
				if(endDist <= r+self.eps):
					print("endDist Il apendeaza pe "+str(objs[endIdx])+" endDist "+str(endDist))
					helper1 = deepcopy(objs[endIdx])
					winG.append(helper1)
				endIdx = endIdx - 1
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
				
			while(startDist <= r and startIdx < endIdx):
				print("AL DOILEA WHILE startIdx = "+str(startIdx)+" obj "+str(objs[startIdx])+" start "+str(startDist))
				if(startDist >= r-self.eps):
					print("startDist Il apendeaza pe "+str(objs[startIdx])+" endDist "+str(startDist))
					helper2 = deepcopy(objs[startIdx])
					winL.append(helper2)
				startIdx = startIdx + 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				
			if(startIdx < endIdx):
				print("Pentru "+str(objs[endIdx])+" endDist este "+str(endDist)+" iar r-eps este "+str(r-self.eps))
				if(endDist >= r-self.eps):
					print("endDist Il apendeaza pe "+str(objs[endIdx])+" endDist "+str(endDist))
					helper3 = deepcopy(objs[endIdx])
					winL.append(helper3)
				if(startDist <= r+self.eps):
					print("startDist Il apendeaza pe "+str(objs[startIdx])+" startDist "+str(startDist))
					helper4 = deepcopy(objs[startIdx])
					winG.append(helper4)
				#exchange items
				print("Before swap start end "+str(objs[startIdx])+" "+str(objs[endIdx])+" "+str(startIdx)+" "+str(endIdx))
				#objs[startIdx], objs[endIdx] = self.swapper(objs[endIdx], objs[startIdx])
				objs[startIdx], objs[endIdx] = objs[endIdx], objs[startIdx]
				print("After swap start end "+str(objs[startIdx])+" "+str(objs[endIdx])+" "+str(startIdx)+" "+str(endIdx))
				startIdx = startIdx + 1
				endIdx = endIdx - 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
		if(startIdx == endIdx):
			print("sunt egale "+str(startIdx)+" "+str(endIdx))
			if(endDist > r and endDist <= r+self.eps):
				helper5 = deepcopy(objs[endIdx])
				winG.append(helper5)
			if(startDist <= r and startDist >= r-self.eps):
				helper6 = deepcopy(objs[startIdx])
				winL.append(helper6)

			if(endDist > r):
				endIdx = endIdx - 1

		print("========================= WIN L ===================")
		print(winL)
		print("========================= WIN L END ===================")

		print("========================= WIN G ===================")
		print(winG)
		print("========================= WIN G END ===================")

		print("========================= OBJS first half ===================")
		print(objs[0:endIdx])
		print("========================= OBJS first half END ===================")

		print("========================= OBJS last half ===================")
		print(objs[endIdx:len(objs)])
		print("========================= OBJS last half END ===================")

		#create partL and partG relative to the distance from p1
		for obj in objs:
			if(self.euclideanDistPosition(obj, p1) < r):
				partL.append(obj)
			else:
				partG.append(obj)

		#return (objs[0:endIdx], objs[endIdx:len(objs)], winL, winG)
		return (partL, partG, winL, winG)

	def quickJoin(self, objs, constSmallNumber):
		objs = list(set(objs))

		print("quick len(objs), constSmallNumber "+str(len(objs))+" "+str(constSmallNumber))
		if(len(objs) == 0):
			return
		if(len(objs) < constSmallNumber):
			#print("GATA! len(objs) "+str(len(objs)))
			self.nestedLoop(objs)
			return

		p1 = self.randomObject(objs)
		#p1 = objs.max(axis=0)
		#p1 = self.centeroidnp(objs)
		
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
		print("Intra in win")

		if (len(objs1) == 0 or len(objs2) == 0):
			print("Una dintre partitii e 0, n-am de ce sa continui")
			return

		if (len(objs1) == 1 or len(objs2) == 1):
			print("Una dintre partitii e 1, n-am de ce sa continui")
			self.nestedLoop2(objs1, objs2)
			return

		totalLen = len(objs1) + len(objs2)
		print("win len(objs), len(objs1), len(objs2), constSmallNumber "+str(totalLen)+" | "+str(objs1)+" | "+str(objs2)+" | "+str(constSmallNumber))

		if(totalLen < constSmallNumber):
			#print("GATA Win! len(objs) "+str(totalLen))
			self.nestedLoop2(objs1, objs2, ["objs1", "objs2"])
			return

		allObjects = objs1 + objs2

		p1 = self.randomObject(allObjects)
		#p1 = allObjects.max(axis=0)
		#p1 = self.centeroidnp(allObjects)

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
					self.allPairs.append([[coord1[0], coord1[1]], [coord2[0], coord2[1]]])

	def nestedLoop2(self, objs1, objs2, deUndeVine = []):
		if(len(deUndeVine) != 0):
			print("nested loop vine din ")
		print("nested loop objs1, objs2 "+ str(objs1) + str(objs2))
		for coord1 in objs1:
			for coord2 in objs2:
				if( self.euclideanDistPosition(coord1, coord2) <= self.eps and coord1 != coord2):
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord1[0], coord1[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.upsertPixelValue("quickDBSCAN",{"$or":[ {"bucket":[]},{"bucket": [coord2[0], coord2[1]] }] }, [[coord1[0], coord1[1]], [coord2[0], coord2[1]]])
					self.allPairs.append([[coord1[0], coord1[1]], [coord2[0], coord2[1]]])

	def upsertPixelValue(self, collection, filter, epsNeigh):
		self.mongoConnectInstance.update(collection, filter, {"$addToSet":{"bucket":{"$each":epsNeigh}}}, True, True)

	def findAndMerge(self, collection, coord):
		#aggregate the results
		if(self.mongoConnectInstance.count(collection, {"bucket": [coord[0], coord[1]] } ) <= 1):
			return
		aggregationString=[{"$match": {"bucket": [coord[0], coord[1]] } },{"$unwind": "$bucket"},{"$group" : {"_id" : ObjectId(), "bucket":{"$addToSet":"$bucket"}}}]
		aggregationResult = self.mongoConnectInstance.aggregate(collection, aggregationString)
		aggregationResultList = list(aggregationResult)
		
		print("Aggregation")
		#remove all other documents - we aggregated them
		print("Count before remove: "+str(self.mongoConnectInstance.count(collection, {})))
		self.mongoConnectInstance.remove(collection, {"bucket": [coord[0], coord[1]] })
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
			color = np.random.rand(3,)
			for pair in document["bucket"]:
				plt.scatter(pair[0], pair[1], c=color)
				#plt.text(pair[0], pair[1], str(pair[0])+', '+str(pair[1]))
				
		print("==================== ALL PAIRS ====================")
		print(self.allPairs)
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
			dataset.append( ( round(float(row[0]), 3), round(float(row[1]), 3), 0) )
			datasetQuick.append( ( round(float(row[0]), 3), round(float(row[1]), 3) ) )

	dataset = np.array( list( set (dataset) ))
	#datasetQuick = datasetQuick[0:200]

	#datasetQuick = datasetQuick[0:100]

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


	'''dbscan = DBSCANKDtreeSimilarityJoin(2.53, dataset)

	start = time.time()

	dbscan.buildIndex()

	end = time.time()

	print('DBSCANKdtree buildIndex took '+str(end - start))

	start = time.time()

	dbscan.doDbscan()

	end = time.time()

	print('DBSCANKdtree took '+str(end - start))

	dbscan.plotClusters()'''

	quickDBSCAN = quickDBSCAN(0.05)
	quickDBSCAN.quickJoin(datasetQuick, 10)
	quickDBSCAN.finalFindAndMerge(datasetQuick)
	quickDBSCAN.plotClusters()




			
			





