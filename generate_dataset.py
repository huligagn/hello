import os
import shutil
import yaml
from time import ctime
import cv2

print ctime()

datasetPath = '/home/huligang/data/myVoc/'
dataset_new = 'dataset/'
if os.path.exists(dataset_new):
	shutil.rmtree(dataset_new)
os.system('mkdir ' + dataset_new)

Annotations = datasetPath + 'Annotations/'
JPEGImages = datasetPath + 'JPEGImages/'

count = 0
for anno_file in os.listdir(Annotations):
	if anno_file[-4:] == '.yml':
		with open(Annotations + anno_file) as f:
			f.readline()
			f.readline()

			data = yaml.load(f.read())
			img_name = data['annotation']['filename']
			img = cv2.imread(JPEGImages + img_name)

			if type(data['annotation']['object']) == list:
				for obj in data['annotation']['object']:
					xmin = int(obj['bndbox']['xmin'])
					ymin = int(obj['bndbox']['ymin'])
					xmax = int(obj['bndbox']['xmax'])
					ymax = int(obj['bndbox']['ymax'])
					assert(xmin < xmax)
					assert(ymin < ymax)

					if not os.path.exists(dataset_new + obj['name']):
						os.system('mkdir ' + dataset_new + obj['name'])

					cv2.imwrite(dataset_new + obj['name'] + '/' + str(count) + '.jpg',
						img[ymin:ymax,xmin:xmax])

					count = count + 1

			elif type(data['annotation']['object']) == dict:
				obj = data['annotation']['object']
				xmin = int(obj['bndbox']['xmin'])
				ymin = int(obj['bndbox']['ymin'])
				xmax = int(obj['bndbox']['xmax'])
				ymax = int(obj['bndbox']['ymax'])
				assert(xmin < xmax)
				assert(ymin < ymax)

				if not os.path.exists(dataset_new + obj['name']):
					os.system('mkdir ' + dataset_new + obj['name'])

				cv2.imwrite(dataset_new + obj['name'] + '/' + str(count) + '.jpg',
					img[ymin:ymax,xmin:xmax])

				count = count + 1
			else:
				print 'WTF!!!!!!!!!!!!!!!!!!!!!!!'
				sys.exit()

print ctime()
print '%d images saved.'%count