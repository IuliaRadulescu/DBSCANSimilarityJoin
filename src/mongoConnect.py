# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"


import pymongo

class MongoDBConnector:
	def __init__(self, dbname, host='localhost', port=27017):
		self.host = host
		self.port = port
		self.client = pymongo.MongoClient(host=self.host, port=self.port)
		self.dbname = dbname
		self.db = self.client[self.dbname]

	def closeConection(self):
		self.client.close()


	def getRecords(self, collection, filter={}, projection={}):
		return self.db[collection].find(filter=filter, projection=projection)

	def getRecord(self, collection, filter={}, projection={}):
		return self.db[collection].find_one(filter=filter, projection=projection)

	def update(self, collection, filter={}, update={}, upsert=True):
		self.db[collection].update(spec=filter, document=update, upsert=upsert)
