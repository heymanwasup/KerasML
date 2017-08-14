
from keras.callbacks import EarlyStopping

import os,sys,argparse
from multiprocessing import Pool

from Config import get_cfg,get_parser
import models,utils 


parser = get_parser()
args = parser.parse_args()
    
def train( cfg ):

    #train model
    model = getattr(models,cfg.model)(*cfg.init_model)
    model.summary()

    model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=cfg.metrics)
    stop = EarlyStopping(monitor='val_loss', patience=10,mode='auto')

    model.fit(cfg.train, cfg.train_cat, sample_weight=cfg.train_weight, 
        validation_data=(cfg.val, cfg.val_cat, cfg.val_weight),
        epochs=100, batch_size=32, callbacks=[stop])

    #test model
    score = model.predict(cfg.test, batch_size=32)

    #save model
    model.save_weights('models/model_{0:}_{1:}.h5'.format(cfg.runtag,cfg.train_tag))

    #make plot
    pltname = './plots/roc_' + cfg.runtag + '_' + cfg.train_tag
    utils.plotROC(score, cfg.test_cat, cfg.test_weight, pltname)

    return


def main(category=0, runtag='test'):
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_0',scale=True)  #test_scale
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_0',scale=False) #test 
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_0',scale=False) #test_2.0.0  retry shallow, no scale 
#  cfg = get_cfg(category,runtag,load_data=True,model_name='3_0',scale=True) #test_3.0.1
#  cfg = get_cfg(category,runtag,load_data=True,model_name='4_0',scale=True) #test_4.0.1
#  cfg = get_cfg(category,runtag,load_data=True,model_name='5_0',scale=True) #test_5.0.1
#  cfg = get_cfg(category,runtag,load_data=True,model_name='6_0',scale=True) #test_6.0.1
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_0',scale=True) #test_2.0.1  retry shallow, scale 
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_1',scale=True) #test_2.1.1 softmax 
#  cfg = get_cfg(category,runtag,load_data=True,model_name='2_2',scale=True) #test_2.2.1 signoid, dense 64
  cfg = get_cfg(category,runtag,load_data=True,model_name='2_3',scale=True) #test_2.3.1 signoid, dense 32
  print ('start training on {0:} {1:}'.format(cfg.runtag,cfg.train_tag))
  train( cfg )
  return

if __name__ == '__main__':
  if args.parallel:
    P = Pool(4)
    for n in range(4):
      P.apply_async(main,args=(n,args.runtag,))
    P.close()
    P.join()
    print('finied all categories')
  else:
    runtag = args.runtag
    category = int(args.category)

    main(category, runtag)

