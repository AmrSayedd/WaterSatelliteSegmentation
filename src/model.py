from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Conv2DTranspose,
    Dropout, Input, MaxPooling2D, concatenate
)
from tensorflow.keras.models import Model

def multi_channel_fusion_head(inputs):
    rgb  = inputs[..., :3]
    nir  = inputs[..., 3:4]
    swir = inputs[..., 4:6]
    ndwi = inputs[..., 6:7]

    rgb_feat  = Conv2D(32, 3, padding='same', activation='relu')(rgb)
    nir_feat  = Conv2D(16, 3, padding='same', activation='relu')(nir)
    swir_feat = Conv2D(16, 3, padding='same', activation='relu')(swir)
    ndwi_feat = Conv2D(8,  3, padding='same', activation='relu')(ndwi)

    fused = concatenate([rgb_feat, nir_feat, swir_feat, ndwi_feat], axis=-1)

    fused = Conv2D(64, 1, padding='same', activation='relu')(fused)
    fused = BatchNormalization()(fused)

    return fused

def conv_block2(x, filters):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def build_unet_fusion(input_shape=(128, 128, 7)):

    inputs = Input(input_shape)

    x = multi_channel_fusion_head(inputs)

    # -------- Encoder --------
    c1 = conv_block2(x, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block2(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block2(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    # -------- Bottleneck --------
    c4 = conv_block2(p3, 256)
    c4 = Dropout(0.3)(c4)

    # -------- Decoder --------
    u5 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = conv_block2(u5, 128)

    u6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = conv_block2(u6, 64)

    u7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = conv_block2(u7, 32)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs)

    return model
