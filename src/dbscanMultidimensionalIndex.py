import collections
import csv
import sys
import numpy as np
import time
import rtree.index
from scipy import spatial


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

			
if __name__ == '__main__':

	#read the csv
	datasetFilename = sys.argv[1]
	dataset = list()

	with open(datasetFilename) as csvFile:
		csvReader = csv.reader(csvFile, delimiter=',')
		for row in csvReader:
			dataset.append( (float(row[2]), float(row[3]), 0) )

	dataset = np.array(dataset)

	print(np.shape(dataset))

	dbscan = DBSCANRtree(1, 5, dataset)

	start = time.time()

	dbscan.buildIndex(5, 5)

	end = time.time()

	print('DBSCANRtree buildIndex took '+str(end - start))

	start = time.time()

	dbscan.doDbscan()

	end = time.time()

	print('DBSCANRtree took '+str(end - start))

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




			
			





