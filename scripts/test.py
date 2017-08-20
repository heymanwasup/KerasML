from root_numpy import root2array,rec2array
from functools import *
import numpy as np
import inspect
import ast  

def get_user_attributes(cls):
  boring = dir(type('dummy',(object,),{}))
  return [item for item in inspect.getmembers(cls) if item[0] not in boring ]

def get_boring():
  boring = dir(type('dummy',(object,),{}))
  return [item for item in inspect.getmembers(boring)]
'''
myData = np.genfromtxt("data.txt", names=True)
print myData

newData = np.ndarray()
'''
'''
b = np.hsplit(myData,(2,1))
for _b in b:
  print '-----------------'
  print _b
'''

'''
atrs = get_user_attributes(myData)
atrs = get_boring()
for a in atrs:
  print '--------------------'
  print a
  print '\n\n\n'
'''

def main():
  fname = '/afs/cern.ch/user/c/chenc/public/forChangqiao/splited_traintest_mva24.root'
  branches = ['MET','mBB','dRBB']

  a =  root2array(treename = 'test_sig',selection = '(EventNumberMod2>0.5)&&(EventNumberMod2<1.5)',filenames=fname, branches = branches)


#  data = []
#  for br in ['MET','mBB']:
#    data.append(a[br])

#  out = np.array((map(np.array,zip(*data))))
#  out.dtype = [('MET','<f4'),('mBB','<f4')]

  out = a[branches[0:2]]
  out = rec2array(out)
  print out['MET']
  print out.shape
  print out.dtype
  print out 

main()
