## Theano

## Installation

```
python setup.py install
pip install chainer
```

## Training

* with characters + conv2D

```
python main.py -mode train --model char --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --emb_list emb_list.txt --word_list word_list.txt
```

* with words + conv2D

```
python main.py -mode train --mode word --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --emb_list emb_list.txt --word_list word_list.txt
```

