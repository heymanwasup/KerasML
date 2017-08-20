from __future__ import print_function
from multiprocessing import Pool
from functools import partial,wraps
import json
import os,pickle,argparse,sys

def composer(f):
  def compose(g):
    @wraps(g)
    def composed(*args,**kw):
      return  f(g(*args,**kw))
    return composed
  return compose

def muter(start,end):
  def mute(fun):
    @wraps(fun)
    def muted(*args,**kw):
      print(start)
      stdout, stderr = sys.stdout, sys.stderr
      sys.stdout, sys.stderr = open('/dev/null','w'), open('/dev/null','w')
      res = fun(*args,**kw)
      sys.stdout, sys.stderr = stdout, stderr
      print(end)
      return end
    return muted
  return mute

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--runtag', action='store',  default='test', help='the runtag for training')
  parser.add_argument('-c', '--category', action='store', choices=['-1','0','1','2','3'],help='the train_tag')
  parser.add_argument('-p', '--parallel', action='store_true', help='training parallel in 4 categories')
  return parser


def Launch(fun,args,info=''):
  if args.parallel:
    P = Pool(4)
    for n in range(4):
      P.apply_async(fun,args=(n,args.runtag,))
    P.close()
    P.join()
  else:
    runtag = args.runtag
    category = int(args.category)
    fun(category, runtag)
  print('finished this run',info )

## begin Cfg ##
class Cfg:
  def __init__(self,category,runtag='test'):

    self.runtag = runtag

    if category > 4 or category < 0:
      print ('the category has to be 0-3')
      raise ValueError
    else:
      self.init_cate(category)

    self.scale = True
    self.init_inputs()
    self.init_model()

  def split_arrays(self,sig,bkg,branches,is_weight=False,weight_name=None):

    import numpy as np
    import ast

    cat = np.concatenate(( np.ones(len(sig), dtype=np.uint8), 
          np.zeros(len(bkg), dtype=np.uint8) ))

    _data  = np.concatenate((sig,bkg))

    Ndata     = len(_data[0])
    Nbranches = len(branches)

    data   = _data[branches]
    if is_weight:
      assert Ndata == Nbranches+1
      weight = _data[weight_name]
    else:
      assert Ndata == Nbranches
      weight = None
    
    return data,cat,weight

  def make_inputs_pickle( self ,p_name):
    from root_numpy import root2array
    import numpy as np

    if self.is_weight:
      branches = self.brs + [self.weight_name]
    else:
      branches = self.brs
   
    _root2array = partial( root2array,
        filenames=self.input_file, branches = branches)
    
    sig_train = _root2array(treename=self.t_sig_train,selection=self.train_sel) 
    bkg_train = _root2array(treename=self.t_bkg_train,selection=self.train_sel) 
    sig_test  = _root2array(treename=self.t_sig_test,selection=self.test_sel) 
    bkg_test  = _root2array(treename=self.t_bkg_test,selection=self.test_sel) 

    sig_val,sig_test = np.split(sig_test,[len(sig_test)/2])
    bkg_val,bkg_test = np.split(bkg_test,[len(bkg_test)/2])

    _split_arrays = partial(self.split_arrays,
        branches=self.brs,is_weight=self.is_weight,weight_name=self.weight_name)


    train = _split_arrays(sig_train,bkg_train)
    val   = _split_arrays(sig_val,bkg_val)
    test  = _split_arrays(sig_test,bkg_test)

    with open(p_name,'wb') as f:
      data = {
        'train':train,
        'val'  :val,
        'test' :test,
      }
      pickle.dump(data,f) 



  @muter('Preprocessing inputs...','Finished preprocessing') 
  def preprocess_data(self,write_scaler=True):

    import utils
    from root_numpy import rec2array

    if not self.scale:
      self.train = rec2array(self.train)
      self.test  = rec2array(self.test)
      self.val   = rec2array(self.val)

    else:
      scalers     = utils.getScalers(self.train,self.brs)
      scaleSample = composer(rec2array)(utils.scaleSample)

      self.train   = scaleSample(self.train, scalers)
      self.test    = scaleSample(self.test, scalers)
      self.val     = scaleSample(self.val, scalers)

      json_path = './results/{0:}/'.format(self.runtag)
      json_file = 'scale_{0:}_{1:}.json'.format(self.runtag,self.train_tag)

      scale_info = {}
      for b in self.brs:

        offset = 0. - scalers[b].mean_
        scale  = 1./scalers[b].scale_

        scale_info[b] = {'offset':offset, 'scale':scale}

      if not os.path.isdir(json_path):
        os.system('mkdir -p {0:}'.format(json_path))

      if not os.path.isfile(json_path+json_file):
        with open(json_path+json_file,'w') as f:
          json.dump(scale_info,f,indent=2)
          print (json_path+json_file + '  Generated!')
          
    return

  def fill_data(self):
    p_name = './data/data_{0:}.pickle'.format(self.train_tag)

    try:
      with open( p_name ) as f:
        data = pickle.load(f)
    except:
      self.make_inputs_pickle( p_name ) 
      with open( p_name ) as f:
        data = pickle.load(f)

    self.book_numpy_array(data)

    self.preprocess_data()

  def book_numpy_array(self,data):
    for key,value in data.iteritems():
      setattr(self, key,           value[0])
      setattr(self, key+'_cat',    value[1])
      setattr(self, key+'_weight', value[2])

  def print(self,*args,**kw):
    sys.stdout,current_stdout = self.stdout,sys.stdout
    print(*args,**kw)
    sys.stdout = current_stdout


    
  def initialize_training(self):
    print ('start training on {0:} {1:}'.format(self.runtag,self.train_tag))

    output_dir = './results/{0:}'.format(self.runtag)
    if not os.path.isdir(output_dir):
      print ('dir not exists, mkdir!')
      os.system('mkdir -p {0:}'.format(output_dir))
      
    log = open('{0:}/{1:}.log'.format(output_dir,self.train_tag),'w')
    err = open('{0:}/{1:}.err'.format(output_dir,self.train_tag),'w')

    sys.stdout,self.stdout = log,sys.stdout
    sys.stderr,self.stderr = log,sys.stderr


  def finalize_training(self):
    sys.stdout = self.stdout
    sys.stderr = self.stderr
    print( 'ended training on {0:}, {1:} !'.format(self.runtag,self.train_tag))

    
  def init_inputs(self):
    self.input_file = '/afs/cern.ch/user/c/chenc/public/forChangqiao/splited_traintest_mva24.root'

    self.t_sig_train = 'train_sig'
    self.t_bkg_train = 'train_bkg'
    self.t_sig_test = self.t_sig_train
    self.t_bkg_test = self.t_bkg_train
    
    self.is_weight   = True
    self.weight_name = 'EventWeight'

  def init_model(self):
    self.model = 'model_2_0'
    self.args_model = (len(self.brs),)
    self.optimizer = 'rmsprop'
    self.loss='binary_crossentropy'
    self.metrics=['accuracy']

      
  def init_cate(self,category):
    self.category = category

    brancheses = {
      2:["MET","HT","dPhiMETdijet","pTB1","pTB2","mBB","dRBB","dEtaBB"],
      3:["MET","HT","dPhiMETdijet","pTB1","pTB2","mBB","dRBB","dEtaBB","pTJ3","mBBJ"],
    }
    
    cuts_nj = {
      2:"(nJ>1.5)&&(nJ<2.5)",
      3:"(nJ>2.5)&&(nJ<3.5)",
    }

    cuts_parity = {
      0:"(EventNumberMod2%2>-0.5)&&(EventNumberMod2%2<0.5)",
      1:"(EventNumberMod2%2>0.5)&&(EventNumberMod2%2<1.5)",
    }
    
    trans = { 0:"E", 1:"O" }

    parity = category%2
    nj = (category>>1)%2 + 2
  
    train_tag = '{0:}j{1:}'.format(nj,trans[parity])

    self.train_tag = train_tag
    self.brs = brancheses[nj]
    self.train_sel = '({0:})&&({1:})'.format(cuts_nj[nj],cuts_parity[parity])
    self.test_sel  = '({0:})&&({1:})'.format(cuts_nj[nj],cuts_parity[(parity+1)%2])
## Cfg ended ##

def _get_cfg(category,runtag,load_data,model_name,scale=True,):

  cfg = Cfg(category, runtag)

  cfg.model = 'model_{0:}'.format(model_name)
  cfg.scale = scale

  if load_data:
    cfg.fill_data()

  return cfg

def setw(*args):
  def _setw(cfg):
    cfg.args_model += args
  return _setw

def get_cfg(category,runtag,load_data=False):

  additional_features = {
    'test_parallel':{'model_name':'D','callback':setw(32)},
    
    'D2'  :{'model_name':'D','callback':setw(2)},  
    'D4'  :{'model_name':'D','callback':setw(4)},  
    'D8'  :{'model_name':'D','callback':setw(8)},  
    'D16' :{'model_name':'D','callback':setw(16)},  
    'D32' :{'model_name':'D','callback':setw(32)},  
    'D64' :{'model_name':'D','callback':setw(64)},  
    'D128':{'model_name':'D','callback':setw(128)},  

    'D8D16'   :{'model_name':'D_D','callback':setw(8,16)},
    'D16D32'  :{'model_name':'D_D','callback':setw(16,32)},
    'D32D64'  :{'model_name':'D_D','callback':setw(32,64)},
    'D64D128' :{'model_name':'D_D','callback':setw(64,128)},
      
    
    'D8D8'    :{'model_name':'D_D','callback':setw(8,8)},
    'D16D16'  :{'model_name':'D_D','callback':setw(16,16)},
    'D32D32'  :{'model_name':'D_D','callback':setw(32,32)},
    'D64D64'  :{'model_name':'D_D','callback':setw(64,64)},
    'D128D128':{'model_name':'D_D','callback':setw(128,128)},

    'D8D4'   :{'model_name':'D_D','callback':setw(8,4)},
    'D16D8'  :{'model_name':'D_D','callback':setw(16,8)},
    'D32D16' :{'model_name':'D_D','callback':setw(32,16)},
    'D64D32' :{'model_name':'D_D','callback':setw(64,32)},
    'D128D64':{'model_name':'D_D','callback':setw(128,64)},

    'D8D16D32'    :{'model_name':'D_D_D','callback':setw(8,16,32)},
    'D16D32D64'   :{'model_name':'D_D_D','callback':setw(16,32,64)},
    'D32D64D128'  :{'model_name':'D_D_D','callback':setw(32,64,128)},
    'D64D128D256' :{'model_name':'D_D_D','callback':setw(64,128,256)},
    'D128D256D512':{'model_name':'D_D_D','callback':setw(128,256,512)},
      
    'D8D8D8'      :{'model_name':'D_D_D','callback':setw(8,8,8)},
    'D16D16D16'   :{'model_name':'D_D_D','callback':setw(16,16,16)},
    'D32D32D32'   :{'model_name':'D_D_D','callback':setw(32,32,32)},
    'D64D64D64'   :{'model_name':'D_D_D','callback':setw(64,64,64)},
    'D128D128D128':{'model_name':'D_D_D','callback':setw(128,128,128)},

    'D8D4D2'    :{'model_name':'D_D_D','callback':setw(8,4,2)},
    'D16D8D4'   :{'model_name':'D_D_D','callback':setw(16,8,4)},
    'D32D16D8'  :{'model_name':'D_D_D','callback':setw(32,16,8)},
    'D64D32D16' :{'model_name':'D_D_D','callback':setw(64,32,16)},
    'D128D64D32':{'model_name':'D_D_D','callback':setw(128,64,32)},
    
    'rnn_test'  :{'model_name':'R_D','callback':setw(32,64)},
  }

  assert runtag in additional_features

  callback = additional_features[runtag].pop('callback',None)

  cfg = _get_cfg(category,runtag,load_data,**additional_features[runtag])

  if callback != None:
    callback(cfg)

  return cfg 
 

if __name__ == '__main__':
  pass
