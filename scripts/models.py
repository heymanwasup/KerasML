from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Masking, LSTM, concatenate

def model_2_0(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_2_1(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_2_2(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(16)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_2_3(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(64)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_3_0(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_4_0(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model


def model_5_0(ndim):
    m_in = Input(shape=(ndim,))
    m = Masking(mask_value=-999)(m_in)
    m = LSTM(32)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_6_0(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model
