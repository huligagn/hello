import tensorflow as tf
import os
import scipy.io
import numpy as np
import cv2

USE_TFRECORDS = True

cwd = os.getcwd()

train_set = '../data/hypotheses.tfrecords'

VGG_PATH = cwd + "/../data/imagenet-vgg-verydeep-19.mat"

learning_rate   = 0.0008
batch_size      = 15

img_size        = 227
n_classes       = 20

threshold = tf.constant(0.7, shape=[batch_size, n_classes])


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label_raw': tf.FixedLenFeature([], tf.string),
                                            'img_raw' : tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [batch_size,img_size,img_size,3])
    # img = tf.cast(img, tf.float32)
    img = img* (1. / 255) - 0.5
    label = tf.decode_raw(features['label_raw'], tf.float32)
    # label = label/tf.reduce_sum(label)
    label = tf.reshape(label, [1,n_classes])

    return img, label

# Read TFRecords files for training
train_image, train_label = read_and_decode(train_set)
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
    with tf.name_scope('vgg'):
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


# X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3), name='test_input')
# Y = tf.placeholder(tf.float32, [None, n_classes], name='test_label')
keepratio = tf.placeholder(tf.float32, name='dropout')


        
weights  = {
    'wd1': tf.Variable(tf.truncated_normal([15*15*512, 1024], stddev=0.5), name='wd1'),
    'wd2': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.5), name='wd2')
}
biases   = {
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=0.5), name='bd1'),
    'bd2': tf.Variable(tf.truncated_normal([n_classes], stddev=0.5), name='bd2')
}

def conv_basic(_input, _w, _b, _keepratio):
    with tf.name_scope('my_layers'):
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

    _pred = tf.nn.sigmoid(_pred)

    return _pred

#########################################################
# ...
#########################################################
train_pred = inference(train_image)
train_pred = tf.reduce_max(train_pred, 0, True)

with tf.name_scope('cost'):
    train_cost = tf.nn.l2_loss(train_pred - train_label)
    train_cost = tf.reduce_mean(train_cost)
    tf.scalar_summary("cost", train_cost)

with tf.name_scope('accuracy'):
    tmp = tf.cast(tf.greater(train_pred, threshold),tf.float32)
    _corr = tf.equal(tmp, train_label)
    train_accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
    tf.scalar_summary("accuracy", train_accr)

# # test in feed way
# test_pred = inference(X)
# test_pred = tf.reduce_max(test_pred, 0, True)

# with tf.name_scope('test_cost'):
#     test_cost = tf.nn.l2_loss(test_pred - Y)
#     test_cost = tf.reduce_mean(test_cost)
#     tf.scalar_summary("test_cost", test_cost)

# with tf.name_scope('test_accuracy'):
#     label_pred = tf.cast(tf.greater(test_pred, threshold),tf.float32)
#     _corr = tf.equal(label_pred, Y)
#     test_accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
#     tf.scalar_summary("test_accuracy", test_accr)



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_cost)


# pred = inference(X)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# _corr = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1)) # Count corrects
# accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy

#########################################################
# run the graph
#########################################################

saver = tf.train.Saver({"wd1":weights['wd1'],"wd2":weights['wd2'],"bd1":biases['bd1'],"bd2":biases['bd2']})

with tf.Session() as sess:

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.train.SummaryWriter("./tensorboard/logs", sess.graph) # for 0.8
    merged = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if os.path.exists(os.path.join("model",'new_model.ckpt')):
        print 'loading saved model...'
        saver.restore(sess, os.path.join("model",'new_model.ckpt'))

    max_iteration=10000
    iteration=0

    # x, y = sess.run([batch_train_image, batch_train_label])
    # print x[0].shape
    # cv2.imshow("img",x[0])
    # print y[0]
    # cv2.waitKey(0)

    while iteration<max_iteration:

        _, output_cost, accuracy = sess.run([optimizer, train_cost, train_accr], feed_dict={keepratio:0.7})

        print 'iteration %d cost: %f' % (iteration, output_cost),
        print '\taccuracy: %f' %(accuracy)

        
        if iteration % 50 == 0:

            _, summary, output_cost, accuracy = sess.run([optimizer, merged, train_cost, train_accr], feed_dict={keepratio:0.7})

            print '*******test cost: ',output_cost,
            print '\taccuracy: ', accuracy

            writer.add_summary(summary, iteration) # Write summary
            saver.save(sess, os.path.join('model','new_model.ckpt'))
        iteration+=1

    print 'everything done....'


    coord.request_stop() 
    coord.join(threads)