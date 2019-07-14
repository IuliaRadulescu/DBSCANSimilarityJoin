from sklearn.datasets import make_circles, make_moons, make_blobs
import csv

def datasetWriter(filename, dataset):
	with open(filename, mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for obj in dataset:
			csv_writer.writerow(obj)

def datasetGenerator(n_samples):
	noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
	noisy_moons = make_moons(n_samples=n_samples, noise=.05)
	blobs = make_blobs(n_samples=n_samples, random_state=8)
	return (noisy_circles[0], noisy_moons[0], blobs[0])

# Generate datasets
(noisy_circles_300, noisy_moons_300, blobs_300) = datasetGenerator(300)
(noisy_circles_600, noisy_moons_600, blobs_600) = datasetGenerator(600)
(noisy_circles_1000, noisy_moons_1000, blobs_1000) = datasetGenerator(1000)

#datasetWriter('dataset/noisy_circles_300.csv', noisy_circles_300)
datasetWriter('dataset/noisy_moons_300.csv', noisy_moons_300)
#datasetWriter('dataset/blobs_300.csv', blobs_300)

#datasetWriter('dataset/noisy_circles_600.csv', noisy_circles_600)
datasetWriter('dataset/noisy_moons_600.csv', noisy_moons_600)
#datasetWriter('dataset/blobs_600.csv', blobs_600)

#datasetWriter('dataset/noisy_circles_1000.csv', noisy_circles_1000)
datasetWriter('dataset/noisy_moons_1000.csv', noisy_moons_1000)
#datasetWriter('dataset/blobs_1000.csv', blobs_1000)
