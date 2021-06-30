# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
# model = tf.keras.models.load_model('/content/capstoneproject/Dashboard/dl/xray.h5')


# def xray_pred(path):
#     img = '/content/capstoneproject/Dashboard/' + path
#     img = cv2.imread(img)
#     img = cv2.resize(img,(150,150))
#     img = img.reshape((1,150,150,3))

#     a = model.predict(img)
    
#     return a[0][0]

from xray_model import detect

detect(image=path)