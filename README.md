## POSTagging-benchmark
- Theano
- Chainer

## Data
Penn Treebank
- Training: section 00-18
- Testing: section 23

## Model
char_vecs: chars [1*#chars] -> embed(10) [10*#chars] -> conv2d(w=(50, 50), filter_size=(10,5), stride=(1,1), pad_size=(10,2)) [50*#chars] -> max-pooling2d(filter_size=(10,5), stride=(1,1)) [50*1]
word_vecs: word [1*1] -> embed(100) [100*1]


```julia
function forward(words)
  # char
  chars -> embed -> 

end
```