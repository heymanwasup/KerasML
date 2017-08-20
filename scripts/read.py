import sys

from multiprocessing import Pool
import numpy as np
import models,Config

import ROOT


def read(cfg):
  model = getattr(models,cfg.model)(*cfg.args_model)

  weight_file = './results/{0:}/model_{0:}_{1:}.h5'.format(cfg.runtag,cfg.train_tag) 
  model.load_weights(weight_file)

  test,test_cat,test_weight = cfg.test, cfg.test_cat, cfg.test_weight
  val, val_cat, val_weight = cfg.val, cfg.val_cat, cfg.val_weight

  test = np.concatenate((test,val))
  test_cat = np.concatenate((test_cat,val_cat))
  test_weight = np.concatenate((test_weight,val_weight))
  
  score = model.predict(test,batch_size=32)

  fout = ROOT.TFile('./results/{0:}/rnn_{0:}_{1:}.root'.format(cfg.runtag,cfg.train_tag),"recreate")

  hs = ROOT.TH1F('sig_%s_%s'%(cfg.runtag,cfg.train_tag),'sig_%s_%s'%(cfg.runtag,cfg.train_tag),20000,-0.2,1.2)
  hsn = ROOT.TH1F('sig_%s_%s_nw'%(cfg.runtag,cfg.train_tag),'sig_%s_%s_nw'%(cfg.runtag,cfg.train_tag),20000,-0.2,1.2)

  hb = ROOT.TH1F('bkg_%s_%s'%(cfg.runtag,cfg.train_tag),'bkg_%s_%s'%(cfg.runtag,cfg.train_tag),20000,-0.2,1.2)
  hbn = ROOT.TH1F('bkg_%s_%s_nw'%(cfg.runtag,cfg.train_tag),'bkg_%s_%s_nw'%(cfg.runtag,cfg.train_tag),20000,-0.2,1.2)

  for n,s in enumerate(score):
    if test_cat[n] == 1:
      hs.Fill(s,test_weight[n])
      hsn.Fill(s)
    else:
      assert test_cat[n] == 0
      hb.Fill(s,test_weight[n])
      hbn.Fill(s)

  roc = ROOT.TMVA.ROCCalc(hs,hb).GetROC()
  roc.SetTitle('roc_{0:}_{1:}'.format(cfg.runtag,cfg.train_tag))
  roc.SetName('roc_{0:}_{1:}'.format(cfg.runtag,cfg.train_tag))

  #roc_n = ROOT.TMVA.ROCCalc(hsn,hbn).GetROC()
  #roc_n.SetTitle('roc_{0:}_{1:}_nw'.format(cfg.runtag,cfg.train_tag))
  #roc_n.SetName('roc_{0:}_{1:}_nw'.format(cfg.runtag,cfg.train_tag))

  fout.cd()

  hs.Write()
  hb.Write()
  #hsn.Write()
  #hbn.Write()

  roc.Write()
  #roc_n.Write()
  print ('roc integral %s %s = '%(cfg.runtag,cfg.train_tag),roc.Integral())
  #print ('roc_noweight integral %s %s = '%(cfg.runtag,cfg.train_tag),roc_n.Integral())

  fout.Close()

def main(category, runtag):
  cfg = Config.get_cfg(category,runtag,load_data=True)
  read( cfg )
  return

if __name__ == '__main__':
  parser = Config.get_parser()
  args = parser.parse_args()
  Config.Launch(main, args, '<Read {0:}>'.format(args.runtag))
