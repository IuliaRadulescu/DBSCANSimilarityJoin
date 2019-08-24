from sklearn.datasets import make_circles, make_moons, make_blobs
import csv

def datasetWriter(filename, dataset):
	with open(filename, mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		theObjects = dataset[0]
		theLabels = dataset[1]
		objectsAndLabels = []
		itemSetLength = len(theObjects)

		for itemId in range(itemSetLength):
			obj = []
			theObject = theObjects[itemId]
			for objCoord in theObject:
				obj.append(objCoord)
			objLabel = theLabels[itemId]
			obj.append(objLabel)
		
			csv_writer.writerow(obj)

def datasetGenerator(n_samples):
	(noisy_circles, noisy_circles_labels) = make_circles(n_samples=n_samples, factor=.5, noise=.05)
	(noisy_moons, noisy_moons_labels) = make_moons(n_samples=n_samples, noise=.05)
	(blobs, blobs_labels) = make_blobs(n_samples=n_samples, random_state=8)
	return [(noisy_circles, noisy_circles_labels), (noisy_moons, noisy_moons_labels), (blobs, blobs_labels)]

# Generate datasets
[(noisy_circles_300, noisy_circles_labels_300), (noisy_moons_300, noisy_moons_labels_300), (blobs_300, blobs_labels_300)] = datasetGenerator(300)
[(noisy_circles_600, noisy_circles_labels_600), (noisy_moons_600, noisy_moons_labels_600), (blobs_600, blobs_labels_600)] = datasetGenerator(600)
[(noisy_circles_1000, noisy_circles_labels_1000), (noisy_moons_1000, noisy_moons_labels_1000), (blobs_1000, blobs_labels_1000)] = datasetGenerator(1000)

datasetWriter('dataset/noisy_circles_300_evaluation.csv', (noisy_circles_300, noisy_circles_labels_300))
datasetWriter('dataset/noisy_moons_300_evaluation.csv', (noisy_moons_300, noisy_moons_labels_300))
datasetWriter('dataset/blobs_300_evaluation.csv', (blobs_300, blobs_labels_300))

datasetWriter('dataset/noisy_circles_600_evaluation.csv', (noisy_circles_600, noisy_circles_labels_600))
datasetWriter('dataset/noisy_moons_600_evaluation.csv', (noisy_moons_600, noisy_moons_labels_600))
datasetWriter('dataset/blobs_600_evaluation.csv', (blobs_600, blobs_labels_600))

datasetWriter('dataset/noisy_circles_1000_evaluation.csv', (noisy_circles_1000, noisy_circles_labels_1000))
datasetWriter('dataset/noisy_moons_1000_evaluation.csv', (noisy_moons_1000, noisy_moons_labels_1000))
datasetWriter('dataset/blobs_1000_evaluation.csv', (blobs_1000, blobs_labels_1000))
