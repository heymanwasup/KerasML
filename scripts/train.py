
from keras.callbacks import EarlyStopping

import os,sys,argparse

from Config import *
import models,utils

def train( cfg ):
    print ('1')
    cfg.initialize_training()

    #train model
    model = getattr(models,cfg.model)(*cfg.args_model)
    model.summary()
    print ('1')

    model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=cfg.metrics)
    stop = EarlyStopping(monitor='val_loss', patience=10,mode='auto')

    model.fit(cfg.train, cfg.train_cat, sample_weight=cfg.train_weight, 
        validation_data=(cfg.val, cfg.val_cat, cfg.val_weight),
        epochs=100, batch_size=32, callbacks=[stop])

    #test model
    score = model.predict(cfg.test, batch_size=32)

    #save model
    model.save_weights('./results/{0:}/model_{0:}_{1:}.h5'.format(cfg.runtag,cfg.train_tag))

    #make plot
    pltname = './results/{0:}/roc_{0:}_{1:}'.format(cfg.runtag,cfg.train_tag)
    utils.plotROC(score, cfg.test_cat, cfg.test_weight, pltname)

    cfg.finalize_training()
    return


def main(category,runtag):

  cfg = get_cfg(category,runtag,load_data=True)
  train( cfg )
  return


if __name__ == '__main__':
  parser = get_parser()
  args = parser.parse_args()
  Launch(main, args, '<Train {0:}>'.format(args.runtag))
