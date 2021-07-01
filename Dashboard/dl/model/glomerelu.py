import tensorflow as tf 
import matplotlib.pyplot as plt
import segmentation_models as sm

encoder= 'efficientnetb3'
IMG_SIZE=512


def glomerelupred(path):
    #path='../input/save-images-from-tfrecord-as-png-jpg/107.png'
    # img=tf.image.decode_png(tf.io.read_file(path),channels=3)
    # model=sm.Unet(encoder,input_shape=(IMG_SIZE,IMG_SIZE,3),classes=1)
    # model.load_weights('../input/seg-models/Unet_efficientnetb3_fold_0.h5')

    # plt.imshow(img)
    # plt.imshow((model.predict(img.numpy().reshape((1,512,512,3)))).reshape((512,512,1)),cmap='coolwarm',alpha=0.5)
    # plt.axis('off')
    # plt.show()
    return 0