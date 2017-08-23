setupATLAS
lsetup root
lsetup python
virtualenv --python=python2.7 ve
source ve/bin/activate

if [ "$1" = "keras" ];
then
    echo "installing keras...";
    pip install pip --upgrade
    pip install theano keras h5py sklearn matplotlib tabulate
    pip install --upgrade https://github.com/rootpy/root_numpy/zipball/master
elif [ "$1" = "xgboost" ];
then
    echo "installing xgboost...";
    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost; make -j4; cd ..
elif [ "$1" = "all" ];
then
  echo "installing keras and xgboost...";
  pip install pip --upgrade
  pip install theano keras h5py sklearn matplotlib tabulate
  pip install --upgrade https://github.com/rootpy/root_numpy/zipball/master
  git clone --recursive https://github.com/dmlc/xgboost
  cd xgboost; make -j4; cd ..
else
  echo "ERROR";
fi
