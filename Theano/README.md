## Theano

## Training

* with characters + conv2D

```
python main.py -mode train --model char --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --word_list words.lst --emb_list nyt100.lst
```

* with words + conv2D

```
python main.py -mode train --model word --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --word_list words.lst --emb_list nyt100.lst
```

* with characters + linear + mini-batch

```
python nn_char_batch.py --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --word_list words.lst --emb_list nyt100.lst --batch_size 32
```

* with words + linear + mini-batch

```
python nn_word_batch.py --train_data wsj_00-18.conll --dev_data wsj_22-24.conll --word_list words.lst --emb_list nyt100.lst --batch_size 32
```


