from __future__ import print_function
import sys

from multiprocessing import Pool
import numpy as np
import models,Config

import ROOT

def fill(model,cfg,fout,data,cat,weight,tag):
  score = model.predict(data,batch_size=32)
  hname = '%s_{0:}_{1:}_{2:}'.format(tag,cfg.runtag,cfg.train_tag)

  hs = ROOT.TH1F(hname%('sig'),hname%('sig'),20000,-0.2,1.2)
  hb = ROOT.TH1F(hname%('bkg'),hname%('bkg'),20000,-0.2,1.2)

  for n,s in enumerate(score):
    if cat[n] == 1:
      hs.Fill(s,weight[n])
    else:
      assert cat[n] == 0
      hb.Fill(s,weight[n])

  roc = ROOT.TMVA.ROCCalc(hs,hb).GetROC()
  roc_name = 'roc_{0:}_{1:}_{2:}'.format(tag,cfg.runtag,cfg.train_tag)
  roc.SetTitle(roc_name)
  roc.SetName(roc_name)

  fout.cd()

  hs.Write()
  hb.Write()
  roc.Write()

  print ('roc integral (%s) %s %s = '%(tag,cfg.runtag,cfg.train_tag),roc.Integral())


def read(cfg):
  model = getattr(models,cfg.model)(*cfg.args_model)
  weight_file = './results/{0:}/model_{0:}_{1:}.h5'.format(cfg.runtag,cfg.train_tag) 
  model.load_weights(weight_file)

  train,train_cat,train_weight = cfg.train, cfg.train_cat, cfg.train_weight
  test,test_cat,test_weight = cfg.test, cfg.test_cat, cfg.test_weight
  val, val_cat, val_weight = cfg.val, cfg.val_cat, cfg.val_weight

  test = np.concatenate((test,val))
  test_cat = np.concatenate((test_cat,val_cat))
  test_weight = np.concatenate((test_weight,val_weight))

  fout = ROOT.TFile('./results/{0:}/nn_{0:}_{1:}.root'.format(cfg.runtag,cfg.train_tag),"recreate")

  fill(model,cfg,fout,test,test_cat,test_weight,'test')
  fill(model,cfg,fout,train,train_cat,train_weight,'train')

  fout.Close()

def main(category, runtag):
  cfg = Config.get_cfg(category,runtag,load_data=True)
  read( cfg )
  return

if __name__ == '__main__':
  parser = Config.get_parser()
  args = parser.parse_args()
  Config.Launch(main, args, '<Read {0:}>'.format(args.runtag))
