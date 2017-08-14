from functools import partial
import json
import os,pickle,argparse



def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--runtag', action='store',  default='test', help='the runtag for training')
  parser.add_argument('-c', '--category', action='store', choices=['-1','0','1','2','3'],help='the train_tag')
  parser.add_argument('-p', '--parallel', action='store_true', help='training parallel in 4 categories')
  return parser

def get_cfg(category,runtag,load_data=True,model_name='2_0',scale=True):
  runtag_to_mode = {
    'test_2.0.0':['2_0',False],
    'test_3.0.1':['3_0',True],
    'test_4.0.1':['4_0',True],
    'test_5.0.1':['5_0',True],
    'test_6.0.1':['6_0',True],
  }

  print '1'
  if runtag in runtag_to_mode:
    print '2'
    model_name = runtag_to_mode[runtag][0]
    scale      = runtag_to_mode[runtag][1]


  print '3'
  cfg = Cfg(category, runtag)
  cfg.model = 'model_{0:}'.format(model_name)
  cfg.scale = scale
  if load_data:
    fill_data(cfg)
    if cfg.scale:
      cfg.do_scale(write_scaler=True)
    cfg.reverse_input()
  return cfg


def zip_arrays(sig,bkg,branches,is_weight=False):
  import numpy as np
  cat = np.concatenate((np.ones(len(sig), dtype=np.uint8), np.zeros(len(bkg), dtype=np.uint8)))
  data = np.concatenate((sig,bkg))
  splitted = zip(*data)

  Ndata     = len(data[0])
  Nbranches = len(branches)

  inputs    = {}
  for n,branch in enumerate(branches):
    inputs[branch] = np.array(splitted[n])

  if is_weight:
    if Ndata != Nbranches+1:
      print ('Weighted! {0:} brs, {1:} found (should be {2:})'.format(Nbranches,Ndata,Nbraches+1))
      raise ValueError
    weight = np.array(splitted[Ndata-1])
  else:
    if Ndata != Nbranches:
      print ('No weight! {0:} in data and {1:} in branches'.format(Nbranches,Ndata))
      raise ValueError
    weight = None
  return inputs,cat,weight

def test_data(inputs, cat, weight):
  print ('inputs:')
  for key in inputs:
    print (key,inputs[key][0:5])
  print ('cat:')
  print (cat[0:5],'  ||  ',cat[len(cat)-5:len(cat)])
  print ('weight')
  print (weight[0:5],'  ||  ',weight[len(weight)-5:len(weight)])


def fill_data(cfg,read_pickle=True):
  p_name = './data/data_{0:}.pickle'.format(cfg.train_tag)
  if not read_pickle or not os.path.isfile(p_name):
    get_numpy_array( cfg ) 
  with open('./data/data_{0:}.pickle'.format(cfg.train_tag)) as f:
    data = pickle.load(f)
  paras = data['train'] + data['val'] + data['test']
  cfg.book_numpy_array(*paras)
      
def get_numpy_array( cfg ):

  from root_numpy import root2array
  if cfg.is_weight:
    branches = cfg.brs + [cfg.weight_name]
  else:
    branches = cfg.brs
 
  get_array = partial( root2array,
      filenames=cfg.input_file, branches = branches)
  
  sig_train = get_array(treename=cfg.t_sig_train, selection=cfg.train_sel) 
  bkg_train = get_array(treename=cfg.t_bkg_train, selection=cfg.train_sel) 

  sig_test  = get_array(treename=cfg.t_sig_test, selection=cfg.test_sel) 
  bkg_test  = get_array(treename=cfg.t_bkg_test, selection=cfg.test_sel) 

  sig_val,sig_test = np.split(sig_test,[len(sig_test)/2])
  bkg_val,bkg_test = np.split(bkg_test,[len(bkg_test)/2])

  
  train,train_cat,train_weight = zip_arrays(sig_train,bkg_train,cfg.brs,cfg.is_weight)
  val,val_cat,val_weight = zip_arrays(sig_val,bkg_val,cfg.brs,cfg.is_weight)
  test,test_cat,test_weight = zip_arrays(sig_test,bkg_test,cfg.brs,cfg.is_weight)

  with open('./data/data_{0:}.pickle'.format(cfg.train_tag),'wb') as f:
    data = {
      'train':(train,train_cat,train_weight),
      'val'  :(val,val_cat,val_weight),
      'test' :(test,test_cat,test_weight),
    }
    pickle.dump(data,f) 


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
   
  def do_scale(self,write_scaler=True):
    print ('in do_scale')
    import utils
    scalers = utils.getScalers(self.train,self.brs)
    self.train   = utils.scaleSample(self.train, scalers)
    self.test    = utils.scaleSample(self.test, scalers)
    self.val     = utils.scaleSample(self.val, scalers)
    
    print ('write_scaler = ',write_scaler)
    if write_scaler:
      scale_info = {}
      for b in self.brs:
        offset = 0. - scalers[b].mean_
        scale  = 1./scalers[b].scale_
        scale_info[b] = {'offset':offset, 'scale':scale}
      pickle_file = './data/scale_{0:}_{1:}.pickle'.format(self.runtag,self.train_tag)
      with open(pickle_file,'wb') as f:
        pickle.dump(scale_info,f)
      print (pickle_file + 'generated!')

      json_file = './data/scale_{0:}_{1:}.json'.format(self.runtag,self.train_tag)
      with open(json_file,'w') as f:
        print >>f,json.dumps(scale_info)
      print (json_file + 'generated!')


  def book_numpy_array(self,
      train,train_cat,train_weight,
      val,val_cat,val_weight,
      test,test_cat,test_weight
      ):
    self.train = train
    self.train_cat = train_cat
    self.train_weight = train_weight

    self.test = test
    self.test_cat = test_cat
    self.test_weight = test_weight
    
    self.val = val
    self.val_cat = val_cat
    self.val_weight = val_weight


  def reverse_input_structure(self,name):

    import numpy as np
    tup = []
    for br in self.brs:
      tup.append(getattr(self,name)[br])
    out = np.array((map(np.array,zip(*tup))))
    setattr(self,name,out)
 
    return

  def reverse_input(self):
    self.reverse_input_structure('train')
    self.reverse_input_structure('test')
    self.reverse_input_structure('val')
      

  def init_inputs(self):
    self.input_file = '/afs/cern.ch/user/c/chenc/public/forChangqiao/splited_traintest_mva24.root'
    self.t_sig_train = 'train_sig'
    self.t_bkg_train = 'train_bkg'
    self.t_sig_test = self.t_sig_train
    self.t_bkg_test = self.t_bkg_train
    
    self.is_weight = True
    self.weight_name = 'EventWeight'

  def init_model(self):
    self.model = 'model_shallow'
    self.init_model = (len(self.brs),)
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

  
  
if __name__ == '__main__':
  cfg = Cfg(3)
  fill_data( cfg )
