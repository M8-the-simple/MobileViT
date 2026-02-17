import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers as L

def inverted_residual_block(inputs, num_filters, strides=1, expansion_ratio=1):
    ##point-wise conv
    x = L.Conv2D(filters=expansion_ratio*inputs.shape[-1],
                kernel_size=1,
                padding="same",
                use_bias=False
                )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ##depth-wise conv
    x = L.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ##point-wise conv
    x = L.Conv2D(filters=num_filters,
                 kernel_size=1,
                 padding="same",
                 use_bias=False)(x)
    x = L.BatchNormalization()(x)
    ##Residual connection
    if strides == 1 and (inputs.shape == x.shape):
        return L.Add()([inputs, x]) 
    return x

def mlp(x, mlp_dim, dim, dropout_rate = 0.1):
    x = L.Dense(mlp_dim, activation="swish")(x) ##Ovdje je reka da moremo koristit neku drugu aktivaciju, ali pošto se svugdje (poli inverted residuala) koristi swish, evo zašto ne 
    x = L.Dropout(dropout_rate)(x)                                           
    x = L.Dense(dim)(x)
    x = L.Dropout(dropout_rate)(x)    

    return x                                       

def tranformer_encoder(x, num_heads, dim, mlp_dim):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(num_heads=num_heads, key_dim=dim)(x, x)
    x = L.Add()([x, skip_1]) ##Dodajemo u slučaju da se identity izgubi zbog dubine i pomažu s back propagation-om

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, mlp_dim, dim)
    x = L.Add()([x, skip_2])

    return x

def mobile_vit_block(inputs, num_filters, dim, patch_size=2, num_layers=1):
    B, H, W, C = inputs.shape

    ## 3x3 conv
    x = L.Conv2D(kernel_size=3,
                 filters=C,
                 padding="same",
                 use_bias=False)(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## 1x1 conv
    x = L.Conv2D(kernel_size=1,
                 filters=dim,
                 padding="same",
                 use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    P = patch_size * patch_size 
    N = int(H*W//P) ##Number of patches

    x = L.Reshape((P, N, dim))(x)

    ##Transformer Encoder
    for i in range(num_layers):
        x = tranformer_encoder(x, 1, dim, dim*2)

    ##Reshape it back
    x = L.Reshape((H, W, dim))(x)

    x = L.Conv2D(kernel_size=1,
                 filters=C,
                 padding="same",
                 use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ##Concatenate

    x = L.Concatenate()([x, inputs])
    ## 3x3 convolution
    x = L.Conv2D(kernel_size=3,
                 filters=num_filters,
                 padding="same",
                 use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)


    return x

def MobileVit(input_shape, num_channels, dim, expansion_ratio, num_layers=[2, 4, 3], num_classes=1000):
    ##Inputs
    inputs = L.Input(input_shape)

    ##Stem
    x = L.Conv2D(kernel_size = 3,
                 filters=num_channels[0],
                 strides=2,
                 padding="same", 
                 use_bias=False)(inputs) ##Pošto imamo strides=2, onda ćemo maknuti rubne slučajeve, tj smanjit će nam se dimenzija za input_shape / strides
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)
    x = inverted_residual_block(x, num_channels[1], strides=1, expansion_ratio=expansion_ratio)
    

    ## Stage 1
    x = inverted_residual_block(x, num_channels[2], strides=2, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[3], strides=1, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[4], strides=1, expansion_ratio=expansion_ratio)
    

    ## Stage 2
    x = inverted_residual_block(x, num_channels[5], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[6], dim[0], num_layers=num_layers[0])
    

    ## Stage 3
    x = inverted_residual_block(x, num_channels[7], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[8], dim[1], num_layers=num_layers[1])

    ## Stage 3
    x = inverted_residual_block(x, num_channels[9], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[10], dim[2], num_layers=num_layers[2])
    x = L.Conv2D(filters=num_channels[11],
                 kernel_size=1,
                 padding="same",
                 use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)
    
    ##Classifier
    x = L.GlobalAveragePooling2D()(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

def MobileVit_S(input_shape, num_classes):
    num_channels = [16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dim = [144, 192, 240]
    expansion_ratio = 4
    return MobileVit(input_shape, num_channels, dim, expansion_ratio, num_classes=num_classes)

def MobileVit_XS(input_shape, num_classes):
    num_channels = [16, 32, 48, 48, 48, 64, 96, 80, 120, 96, 144, 384]
    dim = [96, 120, 144]
    expansion_ratio = 4
    return MobileVit(input_shape, num_channels, dim, expansion_ratio, num_classes=num_classes)

def MobileVit_XXS(input_shape, num_classes):
    num_channels = [16, 16, 24, 24, 24, 48, 64, 64, 80, 80, 96, 320]
    dim = [64, 80, 96]
    expansion_ratio = 4
    return MobileVit(input_shape, num_channels, dim, expansion_ratio, num_classes=num_classes)

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    
    model = MobileVit_XXS(input_shape, num_classes=1000)
    model.summary()

