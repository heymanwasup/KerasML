import os,sys

cate = ['2jO','2jE','3jO','3jE']
cmds = 'source trans_to_cpp/cvmfs-setup.sh '
for c in cate:
  tag = '{0:}_{1:}'.format(sys.argv[1],c)
  arc = './data/arch_{0:}.json'.format(tag)
  mod = './models/model_{0:}.h5'.format(tag)
  var = './data/variable_{0:}.json'.format(tag)
  neu = './data/nn_{0:}.json'.format(tag)

  sub_cmd = '&& ./scripts/kerasfunc2json.py {0:} {1:} > {2:} && ./scripts/kerasfunc2json.py {0:} {1:} {2:} > {3:}'.format(arc,mod,var,neu)
  cmds += sub_cmd
os.system(cmds)


