## Theano

## Installation

```
python setup.py install
pip install chainer
```

## Training

* with characters + conv2D + indexing

```
python nn_char.py --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --init_emb nyt.100
```

* with characters + conv2D + character-zero-padding

```
python nn_char_zeropad.py --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --init_emb nyt.100
```

* with characters + word-zero-padding + character-zero-padding

```
python nn_zeropad.py --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --init_emb nyt.100
```

