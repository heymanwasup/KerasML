import xgboost as xgb
import numpy as np
from Config import *

def train(cfg):
  cfg.initialize_training()

  dtrain = xgb.DMatrix(cfg.train, label=cfg.train_cat, weight=cfg.train_weight, missing = -999.0) 
  dtest  = xgb.DMatrix(cfg.test, label=cfg.test_cat, weight=cfg.test_weight, missing = -999.0) 
  dval   = xgb.DMatrix(cfg.val, label=cfg.val_cat, weight=cfg.val_weight, missing = -999.0) 

  evallist = [(dtrain,'train'), (dtest,'test'), (dval,'val')]

  num_round = 10
  bst = xgb.train( cfg.param, dtrain, num_round, evallist , early_stopping_rounds=10)

  model_name = './results/{0:}/%s_{1:}.%s'.format(cfg.runtag,cfg.train_tag)
  bst.save_model(model_name%('model','model'))
  bst.dump_model(model_name%('dump','txt'),'./data/featmap.txt')

  cfg.finalize_training()

def main(category , runtag):
  cfg = get_cfg_xgb(category,runtag,load_data=True)
  train(cfg)
  return 

if __name__ == '__main__':
  parser = get_parser()
  args = parser.parse_args()
  Launch(main, args, '<Train {0:}>'.format(args.runtag))
