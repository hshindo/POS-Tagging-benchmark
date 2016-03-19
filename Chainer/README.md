# Chainer

## Installation

```
python setup.py install
pip install chainer
```

## Training

* with characters

```
python scripts/train.py wsj_00-18.conll words.lst model --test data/wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1 --use-char
```

* without characters

```
python scripts/train.py wsj_00-18.conll words.lst model --test data/wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1
```

* Log file at model/log (execution time at the end of the log file)
* Help: `python scripts/train.py --help`

## Test

```
python scripts/test.py wsj_22-24.conll model/epoch9
```

* Test accuracy and total time will be reported
* Help: `python scripts/test.py --help`
