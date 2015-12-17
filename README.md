## POSTagging-benchmark
- Theano
- Chainer
- Merlin

## Data
Penn Treebank
- Training: section 00-18
- Testing: section 23

## Settings

### Network Structure
```
function forward(words)
  foreach word in words
    char_vectors[i] =
      chars [1*#chars] ->
      embed(10) [10*#chars] ->
      conv2d(linear=(50, 50), filter_size=(10,5), stride=(1,1), pad_size=(10,2)) [50*#chars] ->
      max-pooling2d(filter_size=(10,5), stride=(1,1)) [50*1]
  char_matrix = char_vectors -> concat(2) [50*#words]
  
  word_matrix = words [100*#words] -> embed(100) [100*#words]
  
  sent_matrix = [char_matrix, word_matrix] -> concat(1) [150*#words]
  
  out_matrix =
    sent_matrix ->
    conv2d(linear=(750,300), filter_size=(150,5), stride=(1,1), pad_size=(150,2)) [300*#words] ->
    relu [300*#words] ->
    linear(300,45) [45*#words]
end
```

### Training
- SGD (learning rate: 0.0075)
- #epochs: 10
