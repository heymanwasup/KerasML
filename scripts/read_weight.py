import Config,models
from multiprocessing import Pool
import os

parser = Config.get_parser()
args = parser.parse_args()


def run(cfg):

  model = getattr(models,cfg.model)(*cfg.init_model)

  f_mod = './models/model_{0:}_{1:}.h5'.format(cfg.runtag,cfg.train_tag)
  model.load_weights(f_mod)

  arch = model.to_json()
  tmp = './data/{{0:}}_{0:}_{1:}.json'.format(cfg.runtag,cfg.train_tag)
  with open(tmp.format('temp'),'w') as f:
    f.write(arch)

  os.system('python -m json.tool {0:} > {1:}; rm {0:}'.format(tmp.format('temp'),tmp.format('arch')))

  f_arch = tmp.format('arch')
  f_var  = './data/variable_{0:}_{1:}.json'.format(cfg.runtag,cfg.train_tag)


def main(category=0, runtag='test'):
  cfg = Config.get_cfg(category,runtag)
  print ('start reading on {0:} {1:}'.format(cfg.runtag,cfg.train_tag))
  run( cfg )
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

  cate = ['2jE','2jO','3jE','3jO']
  cmds = 'source trans_to_cpp/cvmfs-setup.sh '
  for nc,c in enumerate(cate):
    tag = '{0:}_{1:}'.format(args.runtag,c)
    arc = './data/arch_{0:}.json'.format(tag)
    mod = './models/model_{0:}.h5'.format(tag)
    var = './data/variable_{0:}.json'.format(tag)
    neu = './data/nn_{0:}.json'.format(tag)
  
    cmds += ' &&  ./scripts/kerasfunc2json.py {0:} {1:} -r {3:} -c {4:} > {2:}'.format(arc,mod,var,args.runtag,int(nc))
    sub_cmd = ' && ./scripts/kerasfunc2json.py {0:} {1:} {2:} > {3:}'.format(arc,mod,var,neu)
    cmds += sub_cmd
  os.system(cmds)
  
