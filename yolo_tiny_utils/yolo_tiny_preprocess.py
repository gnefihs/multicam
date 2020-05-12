import numpy as np
import cv2

def preprocess_batch(images, net_h, net_w):
    n_img = len(images)
    batch_input = np.zeros((n_img, net_h, net_w, 3))
    for i in range(n_img):
        batch_input[i] = np.expand_dims(images[i], 0)
    return batch_input

#not used anymore
def preprocess_input(image, net_h, net_w):
#     new_h, new_w, _ = image.shape

#     # determine the new size of the image
#     if (float(net_w)/new_w) < (float(net_h)/new_h):
#         new_h = (new_h * net_w)//new_w
#         new_w = net_w
#     else:
#         new_w = (new_w * net_h)//new_h
#         new_h = net_h

    # resize the image to the new size
    new_image = cv2.resize(image, (net_w, net_h))/255.

#     # embed the image into the standard letter box
#     new_image = np.ones((net_h, net_w, 3)) * 0.5
#     new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image