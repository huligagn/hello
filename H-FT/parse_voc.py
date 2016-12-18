import numpy as np
import cv2
import yaml
import Image
import tensorflow as tf

# const variable
img_size = 227

rootdir = '/home/huligang/data/myVoc/'
image_path = rootdir + 'JPEGImages/%s.jpg'
anno_path = rootdir + 'Annotations/%s.yml'
train_list = rootdir + 'ImageSets/Main/trainval.txt'
test_list = rootdir + 'ImageSets/Main/test.txt'
classes = {
    "aeroplane":0,
    "bicycle":1,
    "bird":2,
    "boat":3,
    "bottle":4,
    "bus":5,
    "car":6,
    "cat":7,
    "chair":8,
    "cow":9,
    "diningtable":10,
    "dog":11,
    "horse":12,
    "motorbike":13,
    "person":14,
    "pottedplant":15,
    "sheep":16,
    "sofa":17,
    "train":18,
    "tvmonitor":19
}
n_classes = len(classes)



def gen_label(filename):
    one_hot_label = np.zeros(n_classes, dtype=np.float32)
    with open(anno_path % filename) as f:
        f.readline()
        f.readline()
        data = yaml.load(f.read())
        objs = data['annotation']['object']
        if type(objs) == list:
            for obj in objs:
                one_hot_label[classes[obj['name']]] = 1
        else:
            one_hot_label[classes[objs['name']]] = 1

    return one_hot_label

def load_tain_list(train_list):
    # load training list
    with open(train_list) as f:
        lines = f.readlines()
        train_image = []
        train_label = []
        for line in lines:
            line = line.strip()
            train_image.append(image_path % line)
            train_label.append(gen_label(line))

    assert(len(train_image) == len(train_label))
    print 'load train list finished.'

    return train_image, train_label

def load_test_list(test_list):
    # load training list
    with open(test_list) as f:
        lines = f.readlines()
        test_image = []
        test_label = []
        for line in lines:
            line = line.strip()
            test_image.append(image_path % line)
            test_label.append(gen_label(line))

    assert(len(test_image) == len(test_label))
    print 'load test list finished.'

    return test_image, test_label


def create_record(record_name, image, label):
    writer = tf.python_io.TFRecordWriter(record_name)
    for i in xrange(len(image)):
        img = Image.open(image[i])
        img = img.resize((img_size, img_size))
        img_raw = img.tobytes()
        label_raw = label[i].tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()

    print 'saved as ', record_name

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label_raw': tf.FixedLenFeature([], tf.string),
                                            'img_raw' : tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_size,img_size,3])
    img = tf.cast(img, tf.float32)
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label_raw'], tf.float32)
    label = label/tf.reduce_sum(label)
    label = tf.reshape(label, [n_classes])

    return img, label

if __name__ == '__main__':

    train_image,train_label = load_tain_list(train_list)
    test_image,test_label = load_test_list(test_list)


    create_record('../data/train.tfrecords',train_image,train_label)
    create_record('../data/test.tfrecords',test_image,test_label)

    img, label = read_and_decode("../data/train.tfrecords")

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(3):
            im, l = sess.run([img, label])
            print im.shape
            print l

        coord.request_stop()
        coord.join(threads)
