
run_training = True
run_reading  = False

caterory  = 3

class CFG: 
  def __init__(self):
    self.isW = True
    self.runtag = 'split'
    self.input_file = '/afs/cern.ch/user/c/chenc/public/forChangqiao/splited_traintest_mva24.root'
    self.brancheses = [["MET","HT","dPhiMETdijet","pTB1","pTB2","mBB","dRBB","dEtaBB"],["MET","HT","dPhiMETdijet","pTB1","pTB2","mBB","dRBB","dEtaBB","pTJ3","mBBJ"]]

  def set_category(self):
    
if category < 4:
  cate0 = "(nJ>1.5)&&(nJ<2.5)&&(EventNumberMod2%2>-0.5)&&(EventNumberMod2%2<0.5)"; 
  cate1 = "(nJ>1.5)&&(nJ<2.5)&&(EventNumberMod2%2>0.5)&&(EventNumberMod2%2<1.5)";
  cate2 = "(nJ>2.5)&&(nJ<3.5)&&(EventNumberMod2%2>-0.5)&&(EventNumberMod2%2<0.5)";
  cate3 = "(nJ>2.5)&&(nJ<3.5)&&(EventNumberMod2%2>0.5)&&(EventNumberMod2%2<1.5)";
  
  title = [ ['2jE','2jO' ] ,[ '3jE', '3jO']]
  cates = [ [cate0, cate1] ,[ cate2, cate3]]
  
  parity = category%2
  nj = (category>>1)%2
  
  branches = brancheses[nj]
  runtag += title[nj][parity]
  
  sel_train = cates[nj][(parity+0)%2]
  sel_test  = cates[nj][(parity+1)%2]
  print '::: {0:} :::'.format(title[nj][parity])
else:
  print 'the train_verion has to be less than 6'


if run_training:
  import train
  train
if run_training:
  
  
