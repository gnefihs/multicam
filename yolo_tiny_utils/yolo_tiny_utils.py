from time import time
import cv2
import os
import numpy as np
from scipy.special import expit
from yolo_tiny_preprocess import preprocess_batch, preprocess_input
from yolo_tiny_bbox import BoundBox, bbox_iou, draw_boxes

def _sigmoid(x):
    return expit(x)

def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_h
        new_w = (image_w*net_h)/image_h
    
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def decode_netout_w_mask(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    
    boxes = []
    
    # only perform operations on boxes above objectness threshold
    obj_mask = netout[..., 4] > obj_thresh
    
    netout[obj_mask][..., :2]  = _sigmoid(netout[obj_mask][..., :2]) # for x and y
    netout[obj_mask][..., 4]   = _sigmoid(netout[obj_mask][..., 4]) # for objectness score
    netout[obj_mask][..., 5:]  = netout[obj_mask][..., 4][..., np.newaxis] * _softmax(netout[obj_mask][..., 5:]) # obj * sm(class_score)
    netout[obj_mask][..., 5:] *= netout[obj_mask][..., 5:] > obj_thresh # if less than threshold, class_score = 0

    # create meshgrid
    x = np.linspace(0, grid_h-1, grid_h)
    y = np.linspace(0, grid_w-1, grid_w)
    xv, yv = np.meshgrid(x, y)
    
    xywh = netout[...,:4]
    # add "coordinates" of grids to get absolute x and y
    xywh[...,0] += np.expand_dims(xv,-1)#x
    xywh[...,0] /= grid_w

    xywh[...,1] += np.expand_dims(yv,-1)#y
    xywh[...,1] /= grid_h

    # for w and h, need to do it for every anchor box separately
    for ab in range(nb_box):
        xywh[...,2][...,ab] = anchors[2*ab] * np.exp(xywh[...,2][...,ab]) / net_w #w
        xywh[...,3][...,ab] = anchors[2*ab+1] * np.exp(xywh[...,3][...,ab]) / net_w #h

    #filter net output by obj thresh
    myoutput = np.copy(netout[obj_mask])
    
    x = myoutput[:,0]
    y = myoutput[:,1]
    w = myoutput[:,2]
    h = myoutput[:,3]
    #get xmin,max, ymin,max
    boxes_dims = np.swapaxes(np.vstack([x-w/2, y-h/2, x+w/2, y+h/2]),0,1)

    objectness = myoutput[:,4]
    classes = myoutput[:,5:]
    for i,dim in enumerate(boxes_dims):
        box = BoundBox(dim[0], dim[1], dim[2], dim[3], objectness[i], np.array(classes[i]))
        boxes.append(box)
    
    return boxes

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    start_time = time()
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    
    boxes = []
    
    netout[..., :2]  = _sigmoid(netout[..., :2]) # for x and y
    netout[..., 4]   = _sigmoid(netout[..., 4]) # for objectness score
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:]) # obj * softmax(class_score)
    netout[..., 5:] *= netout[..., 5:] > obj_thresh # if less than threshold, class_score = 0

    # create meshgrid
    y = np.linspace(0, grid_h-1, grid_h)
    x = np.linspace(0, grid_w-1, grid_w)
    xv, yv = np.meshgrid(x, y)
    
    xywh = netout[...,:4]
    # add "coordinates" of grids to get absolute x and y
    xywh[...,0] += np.expand_dims(xv,-1)#x
    
    
    xywh[...,0] /= grid_w

    xywh[...,1] += np.expand_dims(yv,-1)#y
    xywh[...,1] /= grid_h

    feature_proc_time = time()
    
    # for w and h, need to do it for every anchor box separately
    for ab in range(nb_box):
        xywh[...,2][...,ab] = anchors[2*ab] * np.exp(xywh[...,2][...,ab]) / net_w #w
        xywh[...,3][...,ab] = anchors[2*ab+1] * np.exp(xywh[...,3][...,ab]) / net_w #h

    #filter net output by obj thresh
    myoutput = np.copy(netout[netout[..., 4] > obj_thresh])
    
    x = myoutput[:,0]
    y = myoutput[:,1]
    w = myoutput[:,2]
    h = myoutput[:,3]
    #get xmin,max, ymin,max
    boxes_dims = np.swapaxes(np.vstack([x-w/2, y-h/2, x+w/2, y+h/2]),0,1)
    
    objectness = myoutput[:,4]
    classes = myoutput[:,5:]
    for i,dim in enumerate(boxes_dims):
        box = BoundBox(dim[0], dim[1], dim[2], dim[3], objectness[i], np.array(classes[i]))
        boxes.append(box)
    
    box_time = time()
    """
    print("feature_proc: %.5f | box_proc: %.5f" % \
    (feature_proc_time - start_time, box_time - feature_proc_time))
    """      
    return boxes


def get_yolo_boxes(model, batch_input, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    start_time = time()
    nb_images, image_h, image_w, _ = images.shape
    """OLD preproc"""
#     #batch_input         = np.zeros((nb_images, net_h, net_w, 3))
#     for i in range(nb_images):
#         batch_input[i] = images[i]*0.00392 # instead of divide by 255
#     preproc_time = time()

    """NEW preproc, takes numpy array images (N*W*H*C)"""
    # normalize 0.003921 ~<= 1/255
    batch_input = images * 0.003921
    preproc_time = time()
    
    
    """Model inference is done here"""
    batch_output = model.predict_on_batch(batch_input)
    inf_time = time()
    batch_boxes  = [None]*nb_images
    
    nms_times = [] #########time
    correct_times = [] #########time
    decode_times = [] #########time
    
    # iterating through each image
    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i]]
        boxes = []
        loopstart_time = time()
        
        # decode the output of every yolo layer
        for j in range(len(yolos)):
            dec_start = time()
            yolo_anchors = anchors[(1-j)*6:(2-j)*6]
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)
            """
            print('decode time for output', yolos[j].shape, time()-dec_start)
            """
            
        decode_time = time() #########time
        decode_times.append(decode_time - loopstart_time)
        
        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        
        correct_time = time() #########time
        correct_times.append(correct_time - decode_time) #########time
        
        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)
        nms_time = time() #########time
        nms_times.append(nms_time - correct_time) #########time

        batch_boxes[i] = boxes
    
    print("prepro: %.4f |net: %.4f |dec: %.4f |box crct: %.4f |nms: %.4f" % \
    (preproc_time - start_time, inf_time - preproc_time, sum(decode_times), sum(correct_times), sum(nms_times)))
        
    return batch_boxes


# OLD DECODING FUNCTION (about x3 times slower)
# def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
#     grid_h, grid_w = netout.shape[:2]
#     nb_box = 3
#     netout = netout.reshape((grid_h, grid_w, nb_box, -1))
#     nb_class = netout.shape[-1] - 5

#     boxes = []

#     netout[..., :2]  = _sigmoid(netout[..., :2])
#     netout[..., 4]   = _sigmoid(netout[..., 4])
#     netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
#     netout[..., 5:] *= netout[..., 5:] > obj_thresh

#     for i in range(grid_h*grid_w):
#         row = i // grid_w
#         col = i % grid_w
        
#         for b in range(nb_box):
#             # 4th element is objectness score
#             objectness = netout[row, col, b, 4]
            
#             if(objectness <= obj_thresh): continue
            
#             # first 4 elements are x, y, w, and h
#             x, y, w, h = netout[row,col,b,:4]

#             x = (col + x) / grid_w # center position, unit: image width
#             y = (row + y) / grid_h # center position, unit: image height
#             w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
#             h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
#             # last elements are class probabilities
#             classes = netout[row,col,b,5:]
            
#             box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

#             boxes.append(box)

#     return boxes
