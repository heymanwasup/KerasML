# KerasML

A machine learning framework based on Keras

## Install virtual environment (only needed in the first time)

```
git clone https://github.com/heymanwasup/KerasML
cd KerasML
source install.sh
```

And since you use "theano" rather than "tensorflow" as default in lxplus, you should modify this file :
```
$HOME/.keras/keras.json

Change "backend": “tensorflow”  to  "backend”:”theano”.
```


### Each time setup on lxplus

```
source setup.sh
```

## Training model
The model structures are wrote in the `scripts/models.py`, user need to chose one of them before lauching the training. 
```
python ./scripts/train.py [-options]
[-p or --parallel:  training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-c <category>:     specify the training category, should be 0~3]
[-r <runtag>:       runtag is a unique str corresponds to this training]
```
The model structure defined at `scripts/models.py` with `Keras functional API (https://keras.io/getting-started/functional-api-guide/)`


## Test model 

This step generate dedicated signal and background distribution and roc curve.
```
python ./scripts/read.py  [-options]
[-p or --parallel:   training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-c <category>:      specify the training category, should be 0~3]
[-r <runtag>:        runtag is a unique str corresponds to this training]
```

## Transform the *.hfd5 model to *.json

This step will generarte the json file which could be used in a c++ framework (https://github.com/heymanwasup/NNReader)
```
python scripts/read_weight.py [-options]
[-p or --parallel:   training parallelly in 4 categories (2/3 jet cross even/odd)] 
[-c <category>:      specify the training category, should be 0~3]
[-r <runtag>:        runtag is a unique str corresponds to this training]
```


