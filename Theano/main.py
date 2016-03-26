import theano

theano.config.floatX = 'float32'


if __name__ == '__main__':
    import argparse
    import train
    import test

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    """ Mode """
    parser.add_argument('-mode', default='train', help='train/test')

    """ Model """
    parser.add_argument('--model', default='char', help='word/char')
    parser.add_argument('--save', type=bool, default=True, help='save model')
    parser.add_argument('--load', type=str, default=None, help='load model')

    """ Data """
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_size', type=int, default=100000000)

    """ Neural Architectures """
    parser.add_argument('--emb_list', default=None, help='initial embedding list file (word2vec output)')
    parser.add_argument('--word_list', default=None, help='initial word list file (word2vec output)')
    parser.add_argument('--w_emb_dim', type=int, default=100, help='dimension of word embeddings')
    parser.add_argument('--c_emb_dim', type=int, default=10, help='dimension of char embeddings')
    parser.add_argument('--w_hidden_dim', type=int, default=300, help='dimension of word hidden layer')
    parser.add_argument('--c_hidden_dim', type=int, default=50, help='dimension of char hidden layer')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')

    """ Training Parameters """
    parser.add_argument('--opt', default='sgd', help='optimization method')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')

    args = parser.parse_args()

    if args.mode == 'train':
        train.train(args=args)
    else:
        test.test(args=args)
