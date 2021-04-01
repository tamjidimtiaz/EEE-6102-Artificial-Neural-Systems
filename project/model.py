def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())

def Unetplusloss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_coefficient(y_true, y_pred))

def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    # f(?,g_height,g_width,inter_channel)
    f = Activation('relu')(add([theta_x, phi_g]))
    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    # rate(?,x_height,x_width)
    # att_x(?,x_height,x_width,x_channel)
    att_x = multiply([x, rate])
    return att_x


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

              padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


def unet_plus(img_w, img_h, n_label, data_format='channels_first'):

    inputs = Input((3, img_w, img_h))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = s
    depth = 4
    features = 16
    C10 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    C10 = BatchNormalization()(C10)
    C11 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C10)
    C11 = BatchNormalization()(C11)
    C12 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C11)
    C12 = BatchNormalization()(C12)
    C13 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C12)
    C13 = BatchNormalization()(C13)
    C14 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C13)
    C14 = BatchNormalization()(C14)
    
    D11 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C10)
    D21 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(D11)
    D31 = Conv2D(8*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(D21)
    D41 = Conv2D(16*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(D31)

    D12 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C11)
    C21 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D11)
    x1 = tf.keras.layers.Concatenate(axis=1)([D12, C21])

    D22 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x1)
    C31 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D21)
    x2 = tf.keras.layers.Concatenate(axis=1)([D22, C31])

    D32 = Conv2D(8*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x2)
    C41 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D31)
    U4 = UpSampling2D(size=(2, 2), data_format=data_format)(D41)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U4)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x3 = tf.keras.layers.Concatenate(axis=1)([D32, C41, CC2]) 


    C22 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x1)
    D13 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C12) 
    x4 = tf.keras.layers.Concatenate(axis=1)([D13, C22])

    C32 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x2)
    D23 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x4) 
    x5 = tf.keras.layers.Concatenate(axis=1)([D23, C32])

    U3 = UpSampling2D(size=(2, 2), data_format=data_format)(x3)
    CC1 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U3)
    CC2 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x6 = tf.keras.layers.Concatenate(axis=1)([D23, C32, CC2]) 

    C23 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x4)
    U2 = UpSampling2D(size=(2, 2), data_format=data_format)(x6)
    CC1 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U2)
    CC2 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    D14 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C13)
    x7 = tf.keras.layers.Concatenate(axis=1)([C23, D14, CC2]) 

    U1 = UpSampling2D(size=(2, 2), data_format=data_format)(x7)
    CC1 = Conv2D(features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U1)
    CC2 = Conv2D(features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x7 = tf.keras.layers.Concatenate(axis=1)([C14, CC2]) 

    C15 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x7)
    conv7 = core.Activation('sigmoid')(C15)
    model = Model(inputs=inputs, outputs=conv7, name='unet_plus')
    model.summary() 
    model.compile(optimizer='adam', loss=Unetplusloss, metrics=dice_coefficient)
    
    return model


def r2_unet_plus(img_w, img_h, n_label, data_format='channels_first'):

    inputs = Input((3, img_w, img_h))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = s
    depth = 4
    features = 16
    C10 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    C10 = BatchNormalization()(C10)
    C11 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C10)
    C11 = BatchNormalization()(C11)
    C12 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C11)
    C12 = BatchNormalization()(C12)
    C13 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C12)
    C13 = BatchNormalization()(C13)
    C14 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C13)
    C14 = BatchNormalization()(C14)
    
    D11 = rec_res_block(C10, 2*features, data_format=data_format)
    D11 = MaxPooling2D((2, 2), data_format=data_format)(D11)
    D21 = rec_res_block(D11, 4*features, data_format=data_format)
    D21 = MaxPooling2D((2, 2), data_format=data_format)(D21)
    D31 = rec_res_block(D21, 8*features, data_format=data_format)
    D31 = MaxPooling2D((2, 2), data_format=data_format)(D31)
    D41 = rec_res_block(D31, 16*features, data_format=data_format)
    D41 = MaxPooling2D((2, 2), data_format=data_format)(D41)

    D12 = rec_res_block(C11, 2*features, data_format=data_format)
    D12 = MaxPooling2D((2, 2), data_format=data_format)(D12)
    C21 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D11)
    x1 = tf.keras.layers.Concatenate(axis=1)([D12, C21])

    D22 = rec_res_block(x1, 4*features, data_format=data_format)
    D22 = MaxPooling2D((2, 2), data_format=data_format)(D22)
    C31 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D21)
    x2 = tf.keras.layers.Concatenate(axis=1)([D22, C31])

    D32 = rec_res_block(x2, 8*features, data_format=data_format)
    D32 = MaxPooling2D((2, 2), data_format=data_format)(D32)
    C41 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D31)
    U4 = UpSampling2D(size=(2, 2), data_format=data_format)(D41)
    U4 = rec_res_block(U4, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U4)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x3 = tf.keras.layers.Concatenate(axis=1)([D32, C41, CC2]) 


    C22 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x1)
    D13 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C12) 
    x4 = tf.keras.layers.Concatenate(axis=1)([D13, C22])

    C32 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x2)
    D23 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x4) 
    x5 = tf.keras.layers.Concatenate(axis=1)([D23, C32])

    U3 = UpSampling2D(size=(2, 2), data_format=data_format)(x3)

    U3 = rec_res_block(U3, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U3)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x6 = tf.keras.layers.Concatenate(axis=1)([D23, C32, CC2]) 

    C23 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x4)
    U2 = UpSampling2D(size=(2, 2), data_format=data_format)(x6)
    U2 = rec_res_block(U2, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U2)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    D14 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C13)
    # D14 = MaxPooling2D((2, 2), data_format=data_format)(D14)
    x7 = tf.keras.layers.Concatenate(axis=1)([C23, D14, CC2]) 

    U1 = UpSampling2D(size=(2, 2), data_format=data_format)(x7)
    U1 = rec_res_block(U1, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U1)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x7 = tf.keras.layers.Concatenate(axis=1)([C14, CC2]) 

    C15 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x7)
    conv7 = core.Activation('sigmoid')(C15)
    model = Model(inputs=inputs, outputs=conv7, name='r2_net_plus')
    model.summary() 
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/content/gdrive/My Drive/Nucleus Data/model.png')
    Image(retina=True, filename='/content/gdrive/My Drive/Nucleus Data/model.png')
    model.compile(optimizer='adam', loss=Unetplusloss, metrics=dice_coefficient)
    
    return model

def att_unet_plus(img_w, img_h, n_label, data_format='channels_first'):

    inputs = Input((3, img_w, img_h))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = s
    depth = 4
    features = 8
    C10 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    C10 = BatchNormalization()(C10)
    C11 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C10)
    C11 = BatchNormalization()(C11)
    C12 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C11)
    C12 = BatchNormalization()(C12)
    C13 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C12)
    C13 = BatchNormalization()(C13)
    C14 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C13)
    C14 = BatchNormalization()(C14)
    
    D11 = MaxPooling2D((2, 2), data_format=data_format)(C10)
    D21 = MaxPooling2D((2, 2), data_format=data_format)(D11)
    D31 = MaxPooling2D((2, 2), data_format=data_format)(D21)
    D41 = MaxPooling2D((2, 2), data_format=data_format)(D31)

    D12 = MaxPooling2D((2, 2), data_format=data_format)(C11)
    C21 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D11)
    x1 = tf.keras.layers.Concatenate(axis=1)([D12, C21])

    D22 = MaxPooling2D((2, 2), data_format=data_format)(x1)
    C31 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D21)
    x2 = tf.keras.layers.Concatenate(axis=1)([D22, C31])

    D32 = MaxPooling2D((2, 2), data_format=data_format)(x2)
    C41 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D31)
    U4 = UpSampling2D(size=(2, 2), data_format=data_format)(D41)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U4)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x3 = tf.keras.layers.Concatenate(axis=1)([D32, C41, CC2]) 


    C22 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x1)
    D13 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C12) 
    x4 = tf.keras.layers.Concatenate(axis=1)([D13, C22])

    C32 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x2)
    D23 = MaxPooling2D((2, 2), data_format=data_format)(x4)
    # D23 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x4) 
    x5 = tf.keras.layers.Concatenate(axis=1)([D23, C32])

    U3 = UpSampling2D(size=(2, 2), data_format=data_format)(x3)
    U3 = attention_block_2d(U3, x2, inter_channel = U3.get_shape().as_list()[1], data_format='channels_first')
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U3)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x6 = tf.keras.layers.Concatenate(axis=1)([D23, C32, CC2]) 

    C23 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x4)
    U2 = UpSampling2D(size=(2, 2), data_format=data_format)(x6)
    U2 = attention_block_2d(U2, x4, inter_channel = U2.get_shape().as_list()[1], data_format='channels_first')
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U2)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    D14 = MaxPooling2D((2, 2), data_format=data_format)(C13)
    # D14 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C13)
    # D14 = MaxPooling2D((2, 2), data_format=data_format)(D14)
    x7 = tf.keras.layers.Concatenate(axis=1)([C23, D14, CC2]) 

    U1 = UpSampling2D(size=(2, 2), data_format=data_format)(x7)
    U1 = attention_block_2d(U1, C13, inter_channel = U1.get_shape().as_list()[1], data_format='channels_first')
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U1)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x7 = tf.keras.layers.Concatenate(axis=1)([C14, CC2]) 

    C15 = Conv2D(n_label, (1, 1), strides = (1,1), padding='same', data_format=data_format)(x7)
    conv7 = core.Activation('sigmoid')(C15)
    model = Model(inputs=inputs, outputs=conv7, name='att_unet_plus')
    model.summary() 
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/content/gdrive/My Drive/Nucleus Data/model.png')
    Image(retina=True, filename='/content/gdrive/My Drive/Nucleus Data/model.png')
    model.compile(optimizer='adam', loss=Unetplusloss, metrics=dice_coefficient)
    
    return model


def att_r2_unet_plus(img_w, img_h, n_label, data_format='channels_first'):

    inputs = Input((3, img_w, img_h))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = s
    depth = 4
    features = 16
    C10 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    C10 = BatchNormalization()(C10)
    C11 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C10)
    C11 = BatchNormalization()(C11)
    C12 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C11)
    C12 = BatchNormalization()(C12)
    C13 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C12)
    C13 = BatchNormalization()(C13)
    C14 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C13)
    C14 = BatchNormalization()(C14)
    
    D11 = rec_res_block(C10, 2*features, data_format=data_format)
    D11 = MaxPooling2D((2, 2), data_format=data_format)(D11)
    D21 = rec_res_block(D11, 4*features, data_format=data_format)
    D21 = MaxPooling2D((2, 2), data_format=data_format)(D21)
    D31 = rec_res_block(D21, 8*features, data_format=data_format)
    D31 = MaxPooling2D((2, 2), data_format=data_format)(D31)
    D41 = rec_res_block(D31, 16*features, data_format=data_format)
    D41 = MaxPooling2D((2, 2), data_format=data_format)(D41)

    D12 = rec_res_block(C11, 2*features, data_format=data_format)
    D12 = MaxPooling2D((2, 2), data_format=data_format)(D12)
    C21 = Conv2D(2*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D11)
    x1 = tf.keras.layers.Concatenate(axis=1)([D12, C21])

    D22 = rec_res_block(x1, 4*features, data_format=data_format)
    D22 = MaxPooling2D((2, 2), data_format=data_format)(D22)
    C31 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D21)
    x2 = tf.keras.layers.Concatenate(axis=1)([D22, C31])

    D32a = rec_res_block(x2, 8*features, data_format=data_format)
    D32 = MaxPooling2D((2, 2), data_format=data_format)(D32a)
    C41 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(D31)
    U4 = UpSampling2D(size=(2, 2), data_format=data_format)(D41)
    U4 = rec_res_block(U4, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U4)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x3 = tf.keras.layers.Concatenate(axis=1)([D32, C41, CC2]) 


    C22 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x1)
    D13 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C12) 
    x4 = tf.keras.layers.Concatenate(axis=1)([D13, C22])

    C32 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x2)
    D23a = rec_res_block(x4, 8*features, data_format=data_format)
    D23 = MaxPooling2D((2, 2), data_format=data_format)(D23a)
    # D23 = Conv2D(4*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(x4) 
    x5 = tf.keras.layers.Concatenate(axis=1)([D23, C32])

    U3 = UpSampling2D(size=(2, 2), data_format=data_format)(x3)
    U3 = attention_block_2d(U3, D32a, inter_channel = U3.get_shape().as_list()[1], data_format='channels_first')
    U3 = rec_res_block(U3, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U3)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x6 = tf.keras.layers.Concatenate(axis=1)([D23, C32, CC2]) 

    C23 = Conv2D(4*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(x4)
    U2 = UpSampling2D(size=(2, 2), data_format=data_format)(x6)
    U2 = attention_block_2d(U2, D23a, inter_channel = U2.get_shape().as_list()[1], data_format='channels_first')
    U2 = rec_res_block(U2, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U2)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    D14a = rec_res_block(C13, 8*features, data_format=data_format)
    D14 = MaxPooling2D((2, 2), data_format=data_format)(D14a)
    # D14 = Conv2D(2*features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C13)
    # D14 = MaxPooling2D((2, 2), data_format=data_format)(D14)
    x7 = tf.keras.layers.Concatenate(axis=1)([C23, D14, CC2]) 

    U1 = UpSampling2D(size=(2, 2), data_format=data_format)(x7)
    U1 = attention_block_2d(U1, D14a, inter_channel = U1.get_shape().as_list()[1], data_format='channels_first')
    U1 = rec_res_block(U1, 4*features, data_format=data_format)
    CC1 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(U1)
    CC2 = Conv2D(8*features, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=data_format)(CC1)
    x7 = tf.keras.layers.Concatenate(axis=1)([C14, CC2]) 

    C15 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x7)
    conv7 = core.Activation('sigmoid')(C15)
    model = Model(inputs=inputs, outputs=conv7, name='att_r2_unet_plus')
    model.summary() 
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/content/gdrive/My Drive/Nucleus Data/model.png')
    Image(retina=True, filename='/content/gdrive/My Drive/Nucleus Data/model.png')
    model.compile(optimizer='adam', loss=Unetplusloss, metrics=dice_coefficient)
    
    return model


model1 = att_r2_unet_plus(128, 128, 1, data_format='channels_first')






checkpointer = tf.keras.callbacks.ModelCheckpoint('att_r2_unet_plus_16.h5', verbose=1, save_best_only=True)
filepath1 = '/content/gdrive/My Drive/Nucleus Data/att_r2_unet_plus_16'

  
  
callbacks1 = [
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath1, monitor="val_loss", verbose=0, save_best_only=True)]


history1 = model1.fit(np.moveaxis(X_train, -1, 1), np.moveaxis(Y_train.astype(float), -1, 1), validation_split=0.1, batch_size=16, epochs=20, callbacks = callbacks1, shuffle=True)

np.save('/content/gdrive/My Drive/Nucleus Data/att_r2_unet_plus/att_r2_unet_plus_16.npy',history1.history)

from keras.models import model_from_json

model_json = model1.to_json()
with open("/content/gdrive/My Drive/Nucleus Data/att_r2_unet_plus/att_r2_unet_plus_16.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("att_r2_unet_plus_16.h5")
print("Saved model to disk")
