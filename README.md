# KerasML

A machine learning framework based on Keras

## Install virtual environment (only needed in the first time)

```
git clone https://github.com/heymanwasup/KerasML
cd KerasML
source install.sh
```

### Each time setup on lxplus

```
source setup.sh
```

## Training model
The model structures are wrote in the `scripts/models.py`, user need to chose one of them during training
```
python ./scripts/train.py [-options]
[-p or --parallel:   training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-r <runtag>:        runtag is a unique str corresponds to this training]
```

## Test model 

This step generate dedicated signal and background distribution and roc curve.
```
python ./scripts/read.py  [-options]
[-p or --parallel:   training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-r <runtag>:        runtag is a unique str corresponds to this training]
```

## Transform the *.hfd5 model to *.json

This step will generarte the json file which could be used in a c++ framework (https://github.com/heymanwasup/NNReader)
```
python scripts/read_weight.py [-options]
[-p or --parallel:   training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-r <runtag>:        runtag is a unique str corresponds to this training]
```
