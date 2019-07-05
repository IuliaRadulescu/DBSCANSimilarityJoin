import csv
import sys
import collections
from random import randint
import numpy as np
import time
#import rtree.index
from scipy import spatial
import pymongo
import mongoConnect


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

	def euclideanDistPosition(self, a, b):
		return np.linalg.norm(a-b)

	def ball_average(self, objs, p1):
		avgDistHelper = []
		for pixel in objs:
			if(pixel!=p1):
				avgDistHelper.append(self.euclideanDistPosition(pixel, p1))
		avgDistHelper = np.array(avgDistHelper)
		return sum(avgDistHelper)/len(avgDistHelper)

	def partition(self, objs, p1):
		partL = [] 
		partG = []
		winL = []
		winG = []
		
		r = self.ball_average(objs, p1)
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
				
		return (objs[0:endIdx], objs[endIdx+1:len(objs)-1], winL, winG)

	def quickJoin(self, objs, constSmallNumber):
		if(len(objs) < constSmallNumber):
			self.nestedLoop(self.eps, objs)
			return;
			
		p1 = self.randomObject(objs)
		
		(partL, partG, winL, winG) = self.partition(objs, p1)
		if(len(winL)>0 and len(winG)>0):
			self.quickJoinWin(self.eps, winL, winG, constSmallNumber)
		if(len(partG)>0):
			self.quickJoin(self.eps, partL, constSmallNumber)
		if(len(partL)>0):
			self.quickJoin(self.eps, partG, constSmallNumber)

	def quickJoinWin(self, objs1, objs2, constSmallNumber):
		print("Intra in win")
		totalLen = len(objs1) + len(objs2)
		if(totalLen < constSmallNumber):
			self.nestedLoop2(self.eps, objs1, objs2)
			return;
		allObjects = objs1 + objs2
		p1 = self.randomObject(allObjects)

		(partL1, partG1, winL1, winG1) = self.partition(objs1, p1)
		(partL2, partG2, winL2, winG2) = self.partition(objs2, p1)

		self.quickJoinWin(winL1, winG2, constSmallNumber)
		self.quickJoinWin(winG1, winL2, constSmallNumber)
		self.quickJoinWin(partL1, partL2, constSmallNumber)
		self.quickJoinWin(partG1, partG2, constSmallNumber)

	def nestedLoop(self, objs):
		for coord1 in objs:
			for coord2 in objs:
				if(coord1 != coord2 and self.euclideanDistPosition(coord1, coord2) <= self.eps):
					#print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":[coord1[0], coord1[1]]}, [coord2[0], coord2[1]])
					self.upsertPixelValue("quickDBSCAN",{"object":[coord2[0], coord2[1]]}, [coord1[0], coord1[1]])


	def nestedLoop2(self, objs1, objs2):
		for coord1 in objs1:
			for coord2 in objs2:
				if(coord1 != coord2 and self.euclideanDistPosition(coord1, coord2) <= self.eps):
					#print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":[coord1[0], coord1[1]]}, [coord2[0], coord2[1]])
					self.upsertPixelValue("quickDBSCAN",{"object":[coord2[0], coord2[1]]}, [coord1[0], coord1[1]])
					


	def upsertPixelValue(self, collection, filter, epsNeigh):
		self.mongoConnectInstance.update("quickDBSCAN", filter, {"$push":{"epsNeighs":epsNeigh}})
			
if __name__ == '__main__':

	sys.setrecursionlimit(15000)

	#read the csv
	datasetFilename = sys.argv[1]
	dataset = list()
	datasetQuick = list()

	with open(datasetFilename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')
		for row in csvReader:
			dataset.append( (float(row[2]), float(row[3]), 0) )
			datasetQuick.append( (float(row[2]), float(row[3]), 0) )

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

	dbscan = DBSCAN(1, 5, dataset)

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

	print('DBSCANKdtree took '+str(end - start))

	quickDBSCAN = quickDBSCAN(1)
	quickDBSCAN.quickJoin(datasetQuick, 10)




			
			





