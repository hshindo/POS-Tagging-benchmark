# Chainer

## Installation

```
python setup.py install
pip install chainer
```

## Training

* with characters

```
python scripts/train.py wsj_00-18.conll words.lst model --test wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1 --use-char
```

* with characters (with character paddings)

```
python scripts/train.py wsj_00-18.conll words.lst model --test wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1 --use-char --pad-char
```

* without characters

```
python scripts/train.py wsj_00-18.conll words.lst model --test wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1
```

* without characters (use linear instead of conv2d)

```
python scripts/train.py wsj_00-18.conll words.lst model --test wsj_22-24.conll --init-emb nyt100.lst --optim SGD 0.0075 --decay-lr --batch 1 --linear-conv
```

* Log file at model/log (execution time at the end of the log file)
* Help: `python scripts/train.py --help`

## Test

```
python scripts/test.py wsj_22-24.conll model/epoch9
```

* Test accuracy and total time will be reported
* Help: `python scripts/test.py --help`
