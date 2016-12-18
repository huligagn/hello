import tensorflow as tf
import os
import scipy.io
import numpy as np
import cv2
from dataset import Dataset

cwd = os.getcwd()

train_set = 'train.txt'
test_set  = 'test.txt'
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"

learning_rate   = 0.001
batch_size      = 15

img_size        = 227
n_classes       = 20

dataset = Dataset(train_set, test_set)


# def read_and_decode(filename):
#     filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename))
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                         features={
#                                             'label_raw': tf.FixedLenFeature([], tf.string),
#                                             'img_raw' : tf.FixedLenFeature([], tf.string)
#                                         })
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [img_size,img_size,3])
#     # img = tf.cast(img, tf.float32)
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.decode_raw(features['label_raw'], tf.float32)
#     # label = label/tf.reduce_sum(label)
#     label = tf.reshape(label, [n_classes])

#     return img, label

# # Read TFRecords files for training
# train_image, train_label = read_and_decode(train_set)
# batch_train_image, batch_train_label = tf.train.shuffle_batch(
#     [train_image, train_label],
#     batch_size=batch_size,
#     num_threads=4,
#     capacity=30000,
#     min_after_dequeue=1000)

# # Read TFRecords files for testing
# test_image, test_label = read_and_decode(test_set)
# batch_test_image, batch_test_label = tf.train.shuffle_batch(
#     [test_image, test_label],
#     batch_size=batch_size,
#     num_threads=4,
#     capacity=10000,
#     min_after_dequeue=2000)

##################################################################
# define model
##################################################################
def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net, mean_pixel
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel



X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))
Y = tf.placeholder(tf.float32, [None, n_classes])
keepratio = tf.placeholder(tf.float32)


        
weights  = {
    'wd1': tf.Variable(tf.truncated_normal([15*15*512, 1024], stddev=0.5)),
    'wd2': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.5))
}
biases   = {
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=0.5)),
    'bd2': tf.Variable(tf.truncated_normal([n_classes], stddev=0.5))
}

def conv_basic(_input, _w, _b, _keepratio):
    # Input
    _input_r = _input
    # Vectorize
    _dense1 = tf.reshape(_input_r, [-1, _w['wd1'].get_shape().as_list()[0]])
    # Fc1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # Fc2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # Return everything
    out = {'input_r': _input_r, 'dense1': _dense1,
        'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out }
    return out

def inference(x):
    net_val, mean_pixel = net(VGG_PATH, x)
    features = tf.reshape(net_val['relu5_4'],[batch_size,-1])

    _pred = conv_basic(features, weights, biases, keepratio)['out']

    # _pred = tf.nn.sigmoid(_pred)

    return _pred

#########################################################
# ...
#########################################################
# pred = inference(batch_train_image)
# cost = tf.nn.l2_loss(pred - batch_train_label)
# cost = tf.reduce_mean(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test_pred = inference(batch_test_image)
# test_cost = tf.nn.l2_loss(test_pred - batch_test_label)
# test_cost = tf.reduce_mean(test_cost)

pred = inference(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
_corr = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy

#########################################################
# run the graph
#########################################################

saver = tf.train.Saver()
saver2 = tf.train.Saver({"wd1":weights['wd1'],"wd2":weights['wd2'],"bd1":biases['bd1'],"bd2":biases['bd2']})

if not os.path.exists('model'):
    os.system('mkdir model')

if not os.path.exists('H-FT/model'):
    os.system('mkdir H-FT/model -p')

with tf.Session() as sess:

    init = tf.initialize_all_variables()
    sess.run(init)

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)

    if os.path.exists(os.path.join("model",'model.ckpt')):
        saver.restore(sess, os.path.join("model",'model.ckpt'))

    max_iteration=1000
    iteration=0

    # x, y = sess.run([batch_train_image, batch_train_label])
    # print x[0].shape
    # cv2.imshow("img",x[0])
    # print y[0]
    # cv2.waitKey(0)

    while iteration<max_iteration:

        x, y = dataset.next_batch(batch_size, 'train')

        _, output_cost, accuracy = sess.run([optimizer, cost, accr], feed_dict={X: x, Y: y, keepratio:0.7})

        print 'iteration %d cost: %f' % (iteration, output_cost),
        print '\titeration %d accuracy: %f' %(iteration, accuracy)
        
        # if iteration>=500 and iteration%50==0:
        if iteration % 5 == 0:
            x, y = dataset.next_batch(batch_size, 'test')
            output_cost, accuracy = sess.run([cost, accr], feed_dict={X: x, Y: y, keepratio:1.})
            print '*******test cost: ',output_cost,
            print '\ttest accuracy: ', accuracy

            saver.save(sess, os.path.join('model','model.ckpt'))
            saver2.save(sess, os.path.join('H-FT/model','model.ckpt'))
        iteration += 1

    print 'everything done....'


    # coord.request_stop() 
    # coord.join(threads)