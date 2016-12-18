import cv2
import numpy as np
import python_ncut_lib as NCut
import parse_voc
import tensorflow as tf
import os
import sys


rootdir = '/home/huligang/data/myVoc/'
image_path = rootdir + 'JPEGImages/%s.jpg'
anno_path = rootdir + 'Annotations/%s.yml'
box_path = rootdir + 'Results/BBoxesB2W8MAXBGR/%s.txt'
train_list = rootdir + 'ImageSets/Main/trainval.txt'
test_list = rootdir + 'ImageSets/Main/test.txt'

batch_size = 15
img_size = 227
num_boxes = 500


def show_boxes(img_name, boxes):
    '''
        input:
            img_name -> path of the image
            boxes -> list of (score,x1,y1,x2,y2)
    '''

    img = cv2.imread(img_name)

    # print 'image shape: ',img.shape

    for ax in boxes:
        cv2.rectangle(img, (int(ax[1]), int(ax[2])), (int(ax[3]), int(ax[4])), (0, 255, 0))

    cv2.imshow("img", img)
    cv2.waitKey(0)

def load_boxes(filename, num):
    '''
        input:
            filename -> the txt file from BING, for example: /home/huligang/data/myVoc/Results/BBoxesB2W8MAXBGR/000001.txt
            num -> the number of the boxes
        ouput:
            boxes -> the list of boxes, for example: (1,1,353,500)
    '''
    boxes = []
    with open(filename) as f:
        total = f.readline()
        lines = f.readlines()
        assert(int(total) > num)
        for i in range(num):
            boxes.append([float(ax) for ax in lines[i].split(',')])

    return boxes

def bounding_box_overlapping(bb1, bb2):
        
    score1, x0_1, y0_1, x1_1, y1_1 = bb1
    w_1 = x1_1 - x0_1 + 1
    h_1 = y1_1 - y0_1 + 1

    score2, x0_2, y0_2, x1_2, y1_2 = bb2
    w_2 = x1_2 - x0_2 + 1
    h_2 = y1_2 - y0_2 + 1

    x0 = max(x0_1,x0_2)
    x1 = min(x1_1,x1_2)
    y0 = max(y0_1,y0_2)
    y1 = min(y1_1,y1_2)
    xover = x1-x0+1
    yover = y1-y0+1

    if xover<=0 or yover<=0:
        return .0

    over = float(xover*yover)
    not_over = w_1*h_1 + w_2 * h_2 - over
    ratio = over / not_over

    return ratio

def gen_affinity_matrix(boxes):
    '''
        input:
            boxes -> boxes
        output:
            W -> the affinity matrix
    '''
    n = len(boxes)
    W = np.eye(n)

    for i in range(n):
        for j in range(n):
            if(i < j):
                W[i, j] = bounding_box_overlapping(boxes[i], boxes[j])
            elif(i > j):
                W[i, j] = W[j, i]

    return W

def get_hypotheses(filename, num_boxes, num_hypotheses):
    '''
        input:
            filename -> the txt file from BING, for example: /home/huligang/data/myVoc/Results/BBoxesB2W8MAXBGR/000001.txt
            num_boxes -> the number of boxes to generate, before Ncut
            num_hypotheses -> the number of cluster
        output:
            hypotheses -> (list)the hypotheses generated
    '''

    hypotheses = []

    boxes = load_boxes(filename,num_boxes)
    W = gen_affinity_matrix(boxes)
    # print 'the shape of affinity matrix: ',W.shape

    eigen_value,vector = NCut.ncut(W,num_hypotheses)
    vec_dis = NCut.discretisation(vector)

    # if vec_dis.getcol(i)

    for i in range(num_hypotheses):
        try:
            hypotheses.append(boxes[vec_dis.getcol(i).indices[0]])
        except:
            print 'WTF!!!!!'
            hypotheses.append(boxes[vec_dis.getcol(i-1).indices[1]])

    return hypotheses


if __name__ == '__main__':
    # image_name = image_path%'000001'
    # txt_name = box_path+'000001.txt'
    # hypotheses = get_hypotheses(txt_name, 100, 5)
    # show_boxes(image_name,hypotheses)

    
    with open(test_list) as f:
        writer = tf.python_io.TFRecordWriter("../data/hypotheses.tfrecords")
        lines = f.readlines()
        for sample in lines:
            print 'writing sample', sample
            sample = sample.strip()

            label = parse_voc.gen_label(sample)

            img = cv2.imread(image_path % sample)
            hypotheses = get_hypotheses(box_path % sample, num_boxes, batch_size)
            images = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)

            for idx, h in enumerate(hypotheses):
                # print idx, h
                tmp = img[int(h[2]):int(h[4]), int(h[1]):int(h[3])]
                images[idx] = cv2.resize(tmp,(img_size, img_size))
            
            img_raw = images.tobytes()
            label_raw = label.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
        writer.close()

