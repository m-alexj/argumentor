'''
Created on Sep 8, 2017

@author: judeaax
'''
import logging
import argparse
from hyperopt import fmin, tpe, hp
import hyperopt.pyll
from hyperopt.pyll import scope
import sys
from basics.index import Index
from basics import loader
import gensim
from keras.optimizers import Nadam
from argm import train
import model.model as M

logger = logging.getLogger("Hyper")
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Performs hyperparameter search.")

parser.add_argument("-t", dest="trainfile", help="The training data.", required=True)
parser.add_argument("-e", dest="devfile", help="The testing data.", required=True)
parser.add_argument("-w", dest="w2vfile", help="The word2vec file. This must end with 'bin' if it is a binary file.", required=True)
parser.add_argument("-maxtoload", dest="maxtoload", help="The number of training data points to load.", type=int, default=-1)
parser.add_argument("-nonnullweight", dest="nonnullweight", help="Weight for non-null classes.", type=float, default=1.0)
parser.add_argument("-p", dest="epochs", help="The number of epochs.", type=int, default=3)

if len(sys.argv) > 1:
    pvals = parser.parse_args(sys.argv[1:])
else:
    base = "/hits/fast/nlp/judeaax/event2/"
    pvals = parser.parse_args("-t!!{0}arg5.train!!-e!!{0}arg5.dev!!-w!!{0}cyb.txt!!-maxtoload!!1000".format(base).split("!!"));  # !!-weights!!out/weights_n

def objective(args):
    embedding_weights, misses = index.makeEmbeddingWeights(N)
    dropout = args[0]
    batchsize = args[1]
    biassize = int(args[2])
    lstmsize = int(args[3])
    cnnfilters = int(args[4])
    print("batchsize: ", batchsize, "biassize: ", biassize, "lstmsize: ", lstmsize, "dropout: ", dropout, "cnnfilters: ", cnnfilters)
    model, _, _ = M.create_model(MAX_LEN, MAX_MENTION_LEN, MAX_CNN_LEN, MAX_SENTENCE, N, N_classes, N_pos, index, embedding_weights, bias_size=biassize, lstm_size=lstmsize, cnn_filters=cnnfilters, dropout=dropout)
    opt = Nadam()
    model.compile(optimizer=opt, loss="kld", metrics=["accuracy"])
    best_f1, best_weights = train(pvals.epochs, int(batchsize), train_points, dev_points, model, f1labels)
    print("best f1: ", best_f1)
    if best_f1 > 0:
        return -best_f1
    else:
        return 1

if __name__ == '__main__':
    w2vfile_binary = pvals.w2vfile.endswith("bin")
    print("Loading word vectors (binary %r)..." % w2vfile_binary, end="", flush=True)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(pvals.w2vfile, binary=w2vfile_binary)
    N = w2v.vector_size
    N_classes = 31 + 1 + 1  # for sequence-final </s> and for sequence-initial <s>
    N_pos = 10
    MAX_LEN = (5 * 2) + 1 + 1 + 1
    MAX_CNN_LEN = 10
    MAX_MENTION_LEN = 1
    MAX_SENTENCE = 40
    
    index = Index(w2v)
    
    train_data = loader.loadData(pvals.trainfile, MAX_LEN)
    dev_data = loader.loadData(pvals.devfile, MAX_LEN)
    loader.buildEventRestrictions(N_classes, index, train_data, dev_data)
    train_points = loader.structureData(train_data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, True, maxToLoad=pvals.maxtoload, nonNullWeight=pvals.nonnullweight, training=True)
    dev_points = loader.structureData(dev_data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, True, maxToLoad=pvals.maxtoload)
    
    f1labels = [x for x in range(N_classes)]
    
    for i in range(N_classes):
        cls = index.getClass(i)
        if cls.lower() in ["null", index.UNKNOWN.lower(), "unknown class"]:
            f1labels.remove(i)
    
    logger.debug("F1 labels: %s" % str([index.getClass(x) for x in f1labels]))

    assert index.getClassIndex("NULL", False) == 1
    
    best = fmin(fn=objective, space=[hp.uniform("dropout", 0, 1), hp.quniform("batchsize", 300, 800, 10), hp.quniform("biassize", 10, 150, 10), hp.quniform("lstmsize", 50, 500, 10), hp.quniform("cnnfilters", 0, 300, 10)], algo=tpe.rand.suggest, max_evals=50)
    
    print(best)
    
