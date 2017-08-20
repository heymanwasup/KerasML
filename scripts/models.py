from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Masking, LSTM, concatenate

def model_D(ndim,nwidth):
    m_in = Input(shape=(ndim,))
    m = Dense(nwidth)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)
   
    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_D_D(ndim,nw1,nw2):
    m_in = Input(shape=(ndim,))
    m = Dense(nw1)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw2)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_D_D_D(ndim,nw1,nw2,nw3):
    m_in = Input(shape=(ndim,))
    m = Dense(nw1)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw2)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw3)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_D_D_D_D(ndim,nw1,nw2,nw3,nw4):
    m_in = Input(shape=(ndim,))
    m = Dense(nw1)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw2)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw3)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nw4)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model



def model_R_D(ndim,nr=32,nd=64):
    print ('in rnn model')
    m_in = Input(shape=(ndim,))
    m = Masking(mask_value=-999)(m_in)
    m = LSTM(nr)(m)
    m = Dense(nd)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    print ('out rnn model')
    return model

