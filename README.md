[![Twitter Follow](https://img.shields.io/twitter/follow/TheShubhanshu.svg?style=social)](https://twitter.com/TheShubhanshu)
[![GitHub license](https://img.shields.io/github/license/napsternxg/DeepSequenceClassification.svg)](https://github.com/napsternxg/DeepSequenceClassification/edit/master/LICENSE)

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

## Preprocessing:

* Currently, we support the preprocessing for the following file formats:
```
<ROOT><DOC>
<DOCNO> DOCUMENT 1 </DOCNO>
     For <TIME TYPE="DATE:DATE">six years</TIME> , <ENTITY TYPE="PERSON">Shubhanshu A. B. Mishra</ENTITY> has made several programming projects after being inspired by <ENTITY TYPE="PERSON">Linus Torvalds</ENTITY>, a very renowned programmer.
</DOC></ROOT>
```
* Each file can contain multiple `DOCNO`. 
* The dir structure consists of many folders of data split for cross validation. It is as follows:
```
data/
data/CV_files
data/CV_files/1/file1.xml
data/CV_files/1/file2.xml
data/CV_files/1/file3.xml
...

data/CV_files/5/file1.xml
data/CV_files/5/file2.xml
```

## Supports:
* Boundary and Category Detection
* Simple RNN and Bidirectional RNN
* Multi task sequence learning (Boundary + Category trained using same model)
* CNN + BRNN

## Coming Up:

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
