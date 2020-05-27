import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

def conv(input_tensor, num_filters):
    encoder = layers.Conv3D(num_filters, kernel_size=[3, 3, 3], padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def conv_block(input_tensor, num_filters):
    encoder = conv(input_tensor,num_filters)
    encoder = conv(encoder, num_filters)
    return encoder

def encoder_block(input_tensor,num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling3D((2,2,2),strides=(2,2,2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor,concat_tensor, num_filters):
    decoder = layers.Conv3DTranspose(num_filters,(2,2,2),strides=(2,2,2),padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor,decoder],axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv3D(num_filters,kernel_size=[3,3,3],padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv3D(num_filters,kernel_size=[3,3,3],padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv3D(num_filters,kernel_size=[3,3,3],padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def output(decoder0):
    with tf.name_scope('outputs'):
        outputs = layers.Conv3D(1, (1,1,1), activation='sigmoid',name='outputs')(decoder0)
        return outputs
        
    

def model(img_shape):
    inputs = layers.Input(shape=img_shape,name='inputs')
    encoder0_pool, encoder0 = encoder_block(inputs, 64)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 128)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 256)
    center = conv(encoder2_pool, 512)
    decoder2 = decoder_block(center, encoder2, 256)
    decoder1 = decoder_block(decoder2, encoder1, 128)
    decoder0 = decoder_block(decoder1, encoder0, 64)
    outputs = output(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model



def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)+losses.mean_squared_error(y_true,y_pred)
    return loss


