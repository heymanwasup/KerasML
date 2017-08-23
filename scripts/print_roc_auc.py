from __future__ import print_function 
from Config import *
import os,sys,argparse
from Config import *
from ROOT import *

def print_auc(runtag,tag):
  print ('\n{0:} {1:}:'.format(tag,runtag))  
  for category in range(4):
    cfg = get_cfg(category,runtag,load_data=False)
    tfile = "./results/{0:}/nn_{0:}_{1:}.root".format(runtag,cfg.train_tag)
    if not os.path.isfile(tfile):
      print ('\n',tfile,'not exists, generate it!\n')
      os.system('python ./scripts/read.py -r {0:} -c {1:}'.format(runtag,category))
    f = TFile(tfile)
    roc = f.Get("roc_{0:}_{1:}_{2:}".format(tag, runtag, cfg.train_tag))
    print ('roc integral %s %s %s = '%(tag, cfg.runtag,cfg.train_tag),roc.Integral())    
  return

def main(runtag):
  print_auc(runtag,'test')
  #print_auc(runtag,'train')

  return

if __name__ == '__main__':
  parser = get_parser()
  args = parser.parse_args()
  main(args.runtag)
