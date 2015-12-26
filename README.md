# Deep Sequence Classification

Generic library for training models for deep neural networks for text sequence classification tasks. 

## Usage:

* Create a json file `config.json` (default name) using the template in `config.json.sample` and specify the parameters for your training. 
* Best strategy is to save the training and test files in vector format in advance and then give their paths in `data_vectors` parameter in the file. 

* Train a model:
```
python model.py --config config_multitask.json --verbose 1
```

* Resume training from saved weights:
```
python model.py --config config_multitask.json --verbose 1 --weights output/models/model_multi_brnn_multitask_h2-45.h5 --base_epochs 45
```

## Supports:
* Boundary and Category Detection
* Simple RNN and Bidirectional RNN
* Multi task sequence learning (Boundary + Category trained using same model)

## Coming Up:
* CNN + BRNN

## Use Cases:
* Named Entity Recognition
* POS Tagging
* Dependency Parsing

## Author:
* Shubhanshu Mishra

## Dependencies:
* Theano
* Keras
* BeautifulSoup (with lxml)
* numpy
* lxml (requires libxml2, libxslt and libxml2-dev)


Install theano and keras using the following commands:
```
pip install --user --upgrade --no-deps git+git://github.com/Theano/Theano.git
pip install --user --upgrade --no-deps git+git://github.com/fchollet/keras.git
```
