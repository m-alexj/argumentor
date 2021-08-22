'''
Argumentor is a program to investigate ACE 2005 event argument extraction.
It was built to show that event arguments can be extracted quite effectively
based on dependency paths connecting triggers and arguments.

This is research code. It is neither too beautiful, nor too efficient.

@author: Alex Judea
'''
import keras
import logging
import argparse
import sys
from basics import loader
from basics.index import Index
from keras.engine.training import Model
import numpy
from basics.points import Points
from sklearn.metrics.classification import f1_score, \
    precision_recall_fscore_support
from sklearn.metrics import classification
from visual import visualize
import pickle
import model.model as M
from sklearn.preprocessing.data import binarize
from tabulate import tabulate
from itertools import groupby
from progressbar.widgets import Bar, Percentage, ETA
from progressbar.bar import ProgressBar
from random import shuffle
from os.path import join, isdir
from posix import mkdir
import gensim
from keras.optimizers import Nadam


logger = logging.getLogger("Argumentor")
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Performs argument extraction training.")
parser.add_argument("-t", dest="trainfile", help="The training data.", required=True)
parser.add_argument("-e", dest="devfile", help="The testing data.", required=True)
parser.add_argument("-v", dest="validationfile", help="The testing data.", required=False)
parser.add_argument("-w", dest="w2vfile", help="The word2vec file. This must end with 'bin' if it is a binary file.", required=True)
parser.add_argument("-b", dest="batch_size", help="The batch size.", type=int, default=128)
parser.add_argument("-lr", dest="lr", help="The learning rate. -1 uses the optimizer's default value.", type=float, default=-1)
parser.add_argument("-p", dest="epochs", help="The number of epochs.", type=int, default=2)
parser.add_argument("-maxtoload", dest="maxtoload", help="The number of training data points to load.", type=int, default=-1)
parser.add_argument("-biassize", dest="biassize", help="The number of dimensions of the bias component (default 150).", type=int, default=150)
parser.add_argument("-lstmsize", dest="lstmsize", help="The number of dimensions of the LSTM components (default 200).", type=int, default=200)
parser.add_argument("-cnnfilters", dest="cnnfilters", help="The number of CNN filters (default 50).", type=int, default=50)
parser.add_argument("-dropout", dest="dropout", help="The dropout probability (default 0.3).", type=float, default=0.3)
parser.add_argument("-nonnullweight", dest="nonnullweight", help="Weight for non-null classes.", type=float, default=1.0)
parser.add_argument("-extension", dest="extension", help="Extension of serialized model weights and visualizations, if any.", type=str, default="n")
parser.add_argument("-weights", dest="weights", help="The weights file. If set, no learning will be performed.", type=str)
parser.add_argument("-outpath", dest="outpath", default="", help="The directory into which all outputs should be written.", type=str, required=False)
parser.add_argument("-tsne", dest="tsne", default=False, help="If the TSNE embeddings should be computed for visualization", type=bool, required=False)

if len(sys.argv) > 1:
    pvals = parser.parse_args(sys.argv[1:])
else:
    # base = "/hits/fast/nlp/judeaax/event2/"
    base = "/data/nlp/judeaax/event2/"
    # base = "/Users/alexjudea/Google Drive/arbeit/"
    pvals = parser.parse_args("-t!!{0}arg5.train!!-e!!{0}arg5.dev!!-w!!{0}cyb.txt!!-p!!4!!-maxtoload!!100!!-nonnullweight!!2.5!!-outpath!!out!!-v!!{0}arg5.test!!-cnnfilters!!0".format(base).split("!!"));  # !!-weights!!out/weights_n

trainfile = pvals.trainfile
devfile = pvals.devfile
w2vfile = pvals.w2vfile
epochs = pvals.epochs
batch_size = pvals.batch_size
validationfile = None
maxtoload = pvals.maxtoload
nonnullweight = pvals.nonnullweight
extension = pvals.extension
weights = pvals.weights
outpath = pvals.outpath
tsne = pvals.tsne

if len(outpath) > 0 and  not isdir(outpath):
    create_dir = input("%s does not exist. Create? (y|n) " % outpath)
    if create_dir == "y":
        mkdir(outpath)

if pvals.validationfile is not None:
    validationfile = pvals.validationfile

MAX_CNN_LEN = 10
MAX_MENTION_LEN = 1
MAX_SENTENCE = 40
MAX_LEN = (5 * 2) + 1 + 1 + 1

if weights:
    logger.debug("Reading weights from %s\nSkipping training." % weights)
else:
    logger.debug("Epochs %d, batch_size %d, max len %d, max CNN len %d, max mention len %d, max sentence length %d, non-null weight %f" % (epochs, batch_size, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, nonnullweight))

w2vfile_binary = w2vfile.endswith("bin")


def evaluate(model: Model, points: Points, batch_size: int):
    '''
    Evaluates points given a model.

    :param model: The model
    :param points: The points
    :param batch_size: A batch size for the model

    Output: predictions, gold responses, raw predictions (distributions over
    classes), raw gold responses
    '''
    p, y = evaluate_input(model, points, batch_size, M.get_input(points),
                          M.get_output(points))
    all_p = numpy.argmax(p, -1)
    all_y = numpy.argmax(y, -1)
    return all_p, all_y, p, y


def evaluate_input(model: Model, points: Points, batch_size: int,
                   _input: [], output: []):
    '''
    Evaluates points given model and input.

    :param model: The model
    :param points: The points
    :param batch_size: A batch size for the model
    :param _input: Input data for the model
    :param output: Gold responses
    '''
    p = model.predict(_input, batch_size, verbose=0)
    y = output

    if isinstance(p, list) or isinstance(y, list):
        # if any of the outputs is a list, we want only the first predictions
        assert len(p) == len(y), "%d %d" % (len(p), len(y))
        for i in range(len(p)):
            assert p[i].shape == y[i].shape, "%s %s" % (str(p[i].shape), str(y[i].shape))
        p = p[0]
        y = y[0]

    return p, y


def do_postprocess(points, p, p_argmax, l):
    null_idx = index.getClassIndex("NULL", False)
    document_groups = groupby(points.TRIGGER_ID, lambda x: x[0])
    for _, value in document_groups:
        indices = numpy.array([x[1] - 1 for x in list(value)], dtype="int32")
        all_pv = p_argmax[indices]
        pv = p[indices]

        for idx in index.class_index.values():

            if idx == null_idx:
                continue
            # here are the IDs corresponding to the class
            matching_idx = numpy.where(all_pv == idx)[0]
            if len(matching_idx) <= 1:
                continue
            probabilities = [pv[x][idx] for x in matching_idx]

            if numpy.std(probabilities) <= l:
                continue

            # get the max
            _max = numpy.argmax(probabilities)

            for i in range(len(matching_idx)):

                if i == _max:
                    continue
                else:
                    p_argmax[indices[matching_idx[i]]] = null_idx
    return p_argmax


def get_f1_and_samples(model:Model, points:Points, batch_size:int, f1labels, postprocess=False, optimize_postprocess=False, l_best=1, f1_weights=None):
    
    argmax_p, argmax_y_val, p, y = evaluate(model, points, batch_size)
    
    prf = precision_recall_fscore_support(argmax_y_val, argmax_p, average="micro", labels=f1labels, sample_weight=f1_weights)
    
    f1_best = prf[2]
    
    if optimize_postprocess:
#         for l in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 3, 3.5]:
        for l in numpy.arange(0.05, 1.0, 0.05):
            argmax_p_copy = numpy.copy(argmax_p)
            argmax_p_copy = do_postprocess(points, p, argmax_p_copy, l)
            o_prf = precision_recall_fscore_support(argmax_y_val, argmax_p_copy, average="micro", labels=f1labels)

            if o_prf[2] > f1_best:
                l_best = l
                f1_best = o_prf[2] 
                logger.debug("optimize postprocess new_best_f1 %.2f original_f1 %.2f new_best_l %f" % (f1_best, prf[2], l))
                logger.debug("Best postprocess l: %f" % l_best)
    
    if postprocess:
        argmax_p = do_postprocess(points, p, argmax_p, l_best)
            
    
#     f1 = f1_score(argmax_y_val, argmax_p, average="micro", labels=f1labels)
    
    prf = precision_recall_fscore_support(argmax_y_val, argmax_p, average="micro", labels=f1labels)
    
    prf_binary = precision_recall_fscore_support(binarize(argmax_y_val.reshape(-1, 1), threshold=1), binarize(argmax_p.reshape(-1, 1), threshold=1), average="binary")
    
    return prf, prf_binary, l_best, argmax_p, argmax_y_val

def confusion_matrix(argmax_p, argmax_y_val, prefix, outpath, printout=False):
        
    classes = [x - 1 for x in range(1, len(index.class_index))]    
    
    target_names = [index.getClass(x + 1) for x in classes]
    
    CM = classification.confusion_matrix(argmax_y_val - 1, argmax_p - 1, labels=classes)
    
    rows = [ ["ROLE"] + target_names]
    rows += [[target_names[x]] + [str(y)for y in CM[x]] for x in range(len(target_names))]
    
    final_string = tabulate(rows, headers="firstrow")
    
    if printout:
        print(final_string)
    
    with open(join(outpath, "%s.confusion.txt" % prefix), "w") as f:
        f.write(final_string)

def classification_report(p, y, points):
        
    target_names = [index.getClass(x) for x in f1labels]
    logger.debug("Target names:%s" % str(target_names))
    
    print(classification.classification_report(y, p, labels=f1labels, target_names=target_names, digits=3))
        
    logger.debug("p.shape in classification_report:%s" % str(p.shape))
    
    for length in range(1, 6):
        indices = numpy.where(points.LENGTH == length)[0]
        p_length = numpy.zeros(len(indices))
        y_length = numpy.zeros(len(indices))
        
        for i in range(len(indices)):
            idx = indices[i]
            p_length[i] = p[idx]
            y_length[i] = y[idx]
        
        non_null = numpy.where(y_length != index.getClassIndex("NULL" , False))[0]
        f1 = f1_score(y_length, p_length, average="micro", labels=f1labels)
        
        print("Length {:d} f1 {:.3f} (suppport total {:d}, support non-null {:d})".format(length, f1, len(p_length), len(non_null)))
    
    null_idx = index.getClassIndex("NULL", False)
    
    # common ancestor paths for False Negative instances
    ca_fns = 0
    # common ancestor paths for False Positive instances
    ca_fps = 0
    # False Negatives
    fns = 0
    # False Positives
    fps = 0
    # total number of paths with common ancestors
    ca_total = 0
    ca_incorrect = 0
    ca_correct = 0
    
    common_ancestor_idx = index.getWordIndex("COMMON_ACESTOR:1", False, lowercase=False)
    
    for i in range(len(p)):
        
        if points.COMMON_ANCESTOR[i] == common_ancestor_idx:
            ca_total += 1
        
        if p[i] != y[i]:
            
            if points.COMMON_ANCESTOR[i] == common_ancestor_idx:
                ca_incorrect += 1
            
            if p[i] == null_idx:
                # FN
                fns += 1
                if points.COMMON_ANCESTOR[i] == common_ancestor_idx:
                    ca_fns += 1
            elif p[i] != null_idx and y[i] == null_idx:
                # FP
                fps += 1
                if points.COMMON_ANCESTOR[i] == common_ancestor_idx:
                    ca_fps += 1
        else :
            if points.COMMON_ANCESTOR[i] == common_ancestor_idx:
                ca_correct += 1
                
    assert ca_correct + ca_incorrect == ca_total
    
    if fps > 0:
        print("Common ancestor false positives / false positives: %.4f" % (ca_fps / fps))
    if ca_total > 0:
        print("Common ancestor false positives / common ancestors: %.4f" % (ca_fps / ca_total))
    else:
        print("No False Positives")
        
    if fns > 0:
        print("Common ancestor false negatives / false negatives: %.4f" % (ca_fns / fns))
    if ca_total > 0:
        print("Common ancestor false negatives / common ancestors: %.4f" % (ca_fns / ca_total))
    else:
        print("No False Negatives")
    
    print("Common ancestors correct / common ancestors: %.4f" % (ca_correct / ca_total))


def debug_log_attrs(c):
    # get the names
    members = [attr for attr in vars(c) if not attr.startswith("__")]
    # print the attributes and their shapes
    for member in members:
        
        if member in ["DOCUMENT_ID", "TRIGGER_ID"]:
            continue
        logger.debug("{:s}\t{:s}".format(member, str(getattr(c, member).shape)))


def get_scores(p, y):
    '''
    Computes true positives, false positives, and false negatives from the predicted and true values. In order to do so, uses sklearn's confusion_matrix.
    :param p: numpy array with predicted values. Note that this has to binarized.
    :param y: numpy array with true values. Note that this has to binarized.
    
    Returns:
        true positives, false positives, false negatives
    '''
    # index i = known group, index j = predicted group
    CM = classification.confusion_matrix(y, p)
    tp = CM[1][1]
    fp = CM[0][1]    
    fn = CM[1][0]
    
    return tp, fp, fn


def _output_significance_scores_for_group(p, y, groups, outpath, file_extention):
    total_tp = 0
    total_fn = 0
    total_fp = 0
    
    with open(join(outpath, "arguments_%s.txt" % file_extention), "w") as f:
    
        for key, value in groups:
            
            indices = [x[1] for x in value]
            
            # get the values for the actual document
            p_v = p[indices]
            y_v = y[indices]
            
            # bring them to the shape expected by sklearn
            p_v = numpy.squeeze(p_v).reshape(-1, 1)
            y_v = numpy.squeeze(y_v).reshape(-1, 1)
            
            # binarize the values; this will conflate multi-class cases where p_v[i] and y_v[i] are both > 1 (non-null), but not of the same value. We compute the number of such cases and count a false positive and a false negative for each of them
            
            indices_of_nonnull_differences = numpy.where((p_v != y_v) & (p_v > 1) & (y_v > 1))[0]
            p_v_diff = p_v[indices_of_nonnull_differences]
            y_v_diff = y_v[indices_of_nonnull_differences]
            
            assert p_v_diff.shape == y_v_diff.shape
                        
            p_v = binarize(p_v, threshold=1.0)
            y_v = binarize(y_v, threshold=1.0)
            
            # sanity assert; true values and predicted values must be of same shape
            assert p_v.shape == y_v.shape
            
            # throw away cases where predicted and true are both zero
            p_non_null = numpy.where(numpy.squeeze(p_v) > 0)[0]
            y_non_null = numpy.where(numpy.squeeze(y_v) > 0)[0]
            
            if len(p_non_null) > 0:
                total_non_null = numpy.unique(numpy.concatenate((p_non_null, y_non_null)))
            else:
                total_non_null = y_non_null
            
            # get only the values we want
            p_v = p_v[total_non_null]
            y_v = y_v[total_non_null]
            
            if len(p_v) == 0 and len(y_v) == 0:
                # if we are left with nothing, move on
                continue
            
            if numpy.array_equal(p_v, y_v):
                    # all values are 1 at this points, according to the logic above, i.e., we have only true positives here
                    assert p_v[0] == 1
                    tp = len(p_v)
                    fp = 0
                    fn = 0
            else:
                # compute the scores if p_v and y_v are not equal
                tp, fp, fn = get_scores(p_v, y_v)
            
            # here, we correct the error described above (in multi-class cases, p_v and y_v may differ for values > 1, which are conflated to 1 when binarized. 
            # Here, we have to account for this. Above, we computed the number of such cases, now he have to correct the respective scores
            tp -= len(p_v_diff)
            fp += len(p_v_diff)
            fn += len(p_v_diff)
            
            total_tp += tp
            total_fn += fn
            total_fp += fp
            
            prec_enum = tp
            prec_denom = (tp + fp)
            
            rec_enum = tp
            rec_denom = (tp + fn)
            
            f.write("%d %d %d %d\n" % (rec_enum, rec_denom, prec_enum, prec_denom))
    
    assert total_fp != 0
    assert total_fn != 0
    
    with open(join(outpath, "arguments_%s_agg.txt" % file_extention), "w") as f:
        f.write("%d %d %d %d" % (total_tp, total_tp + total_fn, total_tp, total_tp + total_fp))


def output_significance_scores(p, y, points, outpath):
    '''
    Writes scores needed to compute significance with the module "ART" (which uses approximate randomization). Writes two files to the output directory, (a) arguments.txt containing scores per document, and (b) arguments_agg.txt containing aggregated scores (summed over documents).
    If the specified arrays contain multi-class classification values, binarizes those values such that for every value v: v_bin  = 1 if v > 1, 0 otherwise.
    :param p: Predicted values
    :param y: True values
    :param points: The points container. Needed to group by document ID
    :param outpath: Output path
    '''
    
    assert len(p) == len(y)
    
    document_groups = groupby(points.DOCUMENT_ID, lambda x: x[0])
    
    _output_significance_scores_for_group(p, y, document_groups, outpath, "document")
    
    sentence_groups = groupby(points.SENTENCE_ID, lambda x: int(x[0]))
    
    _output_significance_scores_for_group(p, y, sentence_groups, outpath, "sentence")

def train(epochs, batch_size, train_points, dev_points, model, f1labels):
    best_f1 = -1.0
    best_weights = None

    logger.info("Training...")

    document_groups = groupby(train_points.SENTENCE_ID, lambda x: x[1])
    index_sequences = []
    indices = []
    batch_counter = 0
    for key, value in document_groups:
        if batch_counter % batch_size == 0:
            if len(indices) > 0:
                index_sequences.append(numpy.array(indices, dtype="int32"))
                indices = []
        for v in (list(value)):
            indices.append(v[1] - 1)
        
        batch_counter += 1
    
    if len(indices) > 0:
        index_sequences.append(numpy.array(indices, dtype="int32"))
        indices = []
    
    avg_params = None
    
    for epoch in range(epochs):
        rs = []
        
        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], max_value=len(index_sequences)).start()
        shuffle(index_sequences)
        counter = 0
        for indices in index_sequences:
            shuffle(indices)
            r = model.train_on_batch(M.get_input_indexed(train_points, indices), M.get_output_indexed(train_points, indices), sample_weight=M.get_weights_indexed(train_points, indices))
            rs.append(r)
            pbar.update(counter)
            counter += 1
        pbar.finish()
        
        rs = numpy.average(rs, 0)
        
        # initially, averaged weights are set to 'None'; in this case, we just adopt the model weights; otherwise, we compute the running average
        avg_params = model.get_weights() if avg_params is None else [0.9 * avg_params[i] + 0.1 * model.get_weights()[i] for i in range(len(avg_params))]
        # before we evaluate on the devset, we store the original weights so that we can continue training afterwards
        original_weights = model.get_weights()
        # then, we set the current averaged weights as the model weights
        model.set_weights(avg_params)
        # and evaluate on the devset
        prf, prf_binary, _, _, _ = get_f1_and_samples(model, dev_points, batch_size, f1labels)
        # finally, we return to the original weights
        model.set_weights(original_weights)
        f1 = prf[2]
        f1_binary = prf_binary[2]
        
        print("\t".join([":".join([str(y) for y in x]) for x in zip(model.metrics_names, rs)]))
        
        print("{:d}\t{:.4f}\t{:.3f}\t{:.3f}".format(epoch + 1, numpy.average(rs), f1, f1_binary))
        
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = avg_params
            logger.info("\t\t=>\tNew best f1:{:.3f}".format(f1))

    return best_f1, best_weights

if __name__ == '__main__':
    
    print("Loading word vectors (binary %r)..." % w2vfile_binary, end="", flush=True)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2vfile, binary=w2vfile_binary)
    N = w2v.vector_size
    N_classes = 31 + 1 + 1  # for sequence-final </s> and for sequence-initial <s>
    N_pos = 10
    
    index = Index(w2v)
    
    train_data = loader.loadData(trainfile, MAX_LEN)
    dev_data = loader.loadData(devfile, MAX_LEN)
    loader.buildEventRestrictions(N_classes, index, train_data, dev_data)
    train_points = loader.structureData(train_data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, True, maxToLoad=maxtoload, nonNullWeight=nonnullweight, training=True)
    dev_points = loader.structureData(dev_data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, True, maxToLoad=maxtoload)
    
    if validationfile is not None:
        logger.info("Preparing validation data...")
        
        validation_data = loader.loadData(validationfile, MAX_LEN)
        validation_points = loader.structureData(validation_data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, False)
    else:
        logger.info("No validation file set.")
        
    embedding_weights, misses = index.makeEmbeddingWeights(N)
    logger.warn("Embedding misses: {:.2f}".format(misses / len(index.word_index)))
    
    debug_log_attrs(train_points)
    
    model, rnn_argmax_model, forward_gru_model = M.create_model(MAX_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N, N_classes, N_pos, index, embedding_weights, bias_size=pvals.biassize, lstm_size=pvals.lstmsize, cnn_filters=pvals.cnnfilters, dropout=pvals.dropout)
    opt = Nadam(lr=pvals.lr if pvals.lr > 0 else 0.002)
    model.compile(optimizer=opt, loss="kld", metrics=["accuracy"])
    

    f1labels = [x for x in range(N_classes)]
    
    for i in range(N_classes):
        cls = index.getClass(i)
        if cls.lower() in ["null", index.UNKNOWN.lower(), "unknown class"]:
            f1labels.remove(i)
    
    logger.debug("F1 labels: %s" % str([index.getClass(x) for x in f1labels]))

    assert index.getClassIndex("NULL", False) == 1

    if not weights:
        # if we didn't set any weights, we have to perform training
        best_f1, best_weights = train(epochs, batch_size, train_points, dev_points, model, f1labels)
        
        weights_file = join(outpath, "weights_%s" % extension) 
        logger.debug("Pickling weights to %s" % weights_file)
        pickle.dump(best_weights, open(weights_file, "wb"))
    else:
        with open(weights, "rb") as pickle_file:
            # here, we just load the weights we have, without retraining
            best_weights = pickle.load(pickle_file)
    # assign the best weights (either newly trained or loaded from file) to the model
    model.set_weights(best_weights)

    prf_val, prf_binary_val, l_best, best_argmax_p_dev, argmax_y_dev = get_f1_and_samples(model, dev_points, batch_size, f1labels, postprocess=False, optimize_postprocess=True)
    logger.info("Final development post-processed micro-averaged p %.3f r %.3f f1 %.3f, final binary p %.3f r %.3f f1 %.3f" % (prf_val[0], prf_val[1], prf_val[2], prf_binary_val[0], prf_binary_val[1], prf_binary_val[2]))
    
#     if isinstance(M.get_argmax_output(dev_points), list):
    p_sequence_argmax, y_sequence_argmax = evaluate_input(rnn_argmax_model, dev_points, batch_size, M.get_argmax_input(dev_points), M.get_argmax_output(dev_points))
#     else:
#         p_sequence_argmax, y_sequence_argmax = (None, None)
    p_forward_gru_model, _ = evaluate_input(forward_gru_model, dev_points, batch_size, M.get_argmax_input(dev_points), M.get_argmax_output(dev_points))
    
    classification_report(best_argmax_p_dev, argmax_y_dev, dev_points)
    confusion_matrix(best_argmax_p_dev, argmax_y_dev, "development", outpath)
    visualize.toSingleHTML(index, dev_points, MAX_LEN, (best_argmax_p_dev, argmax_y_dev), (p_sequence_argmax, y_sequence_argmax), "dev", outpath)
    visualize.toLSTMVis(index, dev_points, forward_gru_model, p_forward_gru_model, p_sequence_argmax, outpath)
    
    if tsne:
        visualize.performTSNEAnalysis(index, dev_points, p_forward_gru_model, argmax_y_dev, N_classes, outpath)
    
    if validationfile is not None:
        logger.info("=== val ===")
        prf_val, prf_binary_val, _, _, _ = get_f1_and_samples(model, validation_points, batch_size, f1labels)
        logger.info("Final validation micro-averaged p %.3f r %.3f f1 %.3f, final binary p %.3f r %.3f f1 %.3f" % (prf_val[0], prf_val[1], prf_val[2], prf_binary_val[0], prf_binary_val[1], prf_binary_val[2]))
        
        prf_val, prf_binary_val, _, best_argmax_p_val, argmax_y_val = get_f1_and_samples(model, validation_points, batch_size, f1labels, postprocess=False, optimize_postprocess=False, l_best=l_best)
        logger.info("Final validation post-processed micro-averaged p %.3f r %.3f f1 %.3f, final binary p %.3f r %.3f f1 %.3f" % (prf_val[0], prf_val[1], prf_val[2], prf_binary_val[0], prf_binary_val[1], prf_binary_val[2]))
        classification_report(best_argmax_p_val, argmax_y_val, validation_points)
        visualize.toSingleHTML(index, validation_points, MAX_LEN, (best_argmax_p_val, argmax_y_val), None, "test", outpath)
        confusion_matrix(best_argmax_p_val, argmax_y_val, "validation", outpath)
        output_significance_scores(best_argmax_p_val, argmax_y_val, validation_points, outpath)
