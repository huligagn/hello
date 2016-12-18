import os

dataset_path = 'dataset'
f_train = open('train.txt', 'w')
f_test = open('test.txt', 'w')

log = []
classes = os.listdir(dataset_path)
for n in range(len(classes)):
	path = os.path.join(dataset_path, classes[n])
	collection = os.listdir(path)
	img_num_per_class = len(collection)
	log.append({classes[n]:img_num_per_class})
	for i in range(img_num_per_class):
		if i < (img_num_per_class / 5):
			f_test.write(os.path.join(path, collection[i])+ ' ' + str(n) + '\n')
		else:
			f_train.write(os.path.join(path, collection[i])+ ' ' + str(n) + '\n')

f_train.close()
f_test.close()

# print classes
for d in log:
	print d