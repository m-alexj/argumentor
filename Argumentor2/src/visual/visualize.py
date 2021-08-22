'''
Created on Aug 17, 2016

@author: Alex Judea
'''

import logging
import numpy
import yaml

import dominate
from dominate.dom_tag import attr
from dominate.tags import span, link, div, br, hr
import h5py
from sklearn import manifold, preprocessing
from basics.index import Index
from basics.points import Points
import matplotlib
from os.path import join
from sklearn.cluster.k_means_ import KMeans
from sklearn.covariance.outlier_detection import EllipticEnvelope
from sklearn.cluster.mean_shift_ import estimate_bandwidth, MeanShift
from model import model
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

logger = logging.getLogger(__name__)


def generateYAML(model, outpath):
    
    type_parts = []
        
    part = dict(type="state", layer="1", path=model.name)
    type_parts.append(part)
    data = dict(
                name="Argumenter",
                description="still developing",
                files=dict(
                             states="states.hdf5",
                             word_ids="train.hdf5",
                             words="words.dict",
#                              entities="entities.hdf5",
#                              entity_dict="entities.dict",
                            label_dict="labels.dict",
                            predicted="predicted.hdf5",
                            actual="actual.hdf5"
                             ),
                word_sequence=dict(
                                     file="word_ids",
                                     path="word_ids",
                                     dict_file="words"
                                     ),
                states=dict(
                              file="states",
                              types=type_parts
                              ),
                meta=dict(
#                           entities=dict(
#                                           file="entities",
#                                           path="entity_ids",
#                                           dict="entity_dict",
#                                           vis=dict(
#                                                    type="discrete",
#                                                    range="dict"
#                                                    )
#                                           ),
                        predicted=dict(
                                       file="predicted",
                                       path="predicted_ids",
                                       dict="label_dict",
                                       vis=dict(
                                                type="discrete",
                                                range="dict"
                                                )
                                       ),
                           
                        actual=dict(
                                       file="actual",
                                       path="actual_ids",
                                       dict="label_dict",
                                       vis=dict(
                                                type="discrete",
                                                range="dict"
                                                )
                                       )
                          )
                )
    with open(join(outpath, 'lstm.yml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        

def toLSTMVis(index, points, model, states, p, outpath):

    states = states.astype("float32")

    if len(states.shape) > 2:
        states = states.reshape(states.shape[0] * states.shape[1], states.shape[2])

    p = p.reshape(p.shape[0] * p.shape[1])

    logger.info("Create layer %s" % model.name)

    f = h5py.File(join(outpath, "states.hdf5"), "w")
    dset3 = f.create_dataset(model.name, states.shape)
    dset3[...] = states

    X_flat = points.X_STATIC + points.X_DYNAMIC
    X_flat = X_flat.reshape(X_flat.shape[0] * X_flat.shape[1])

    for word_id in X_flat.flatten():
        assert word_id in index.inv_word_index, "%d" % word_id

    f = h5py.File(join(outpath, "train.hdf5"), "w")
    dset3 = f.create_dataset('word_ids', X_flat.shape, dtype='int')
    dset3[...] = X_flat
    f = h5py.File(join(outpath, "predicted.hdf5"), "w")
    dset3 = f.create_dataset('predicted_ids', p.shape, dtype='int')
    dset3[...] = p

    f = h5py.File(join(outpath, "actual.hdf5"), "w")
    y = numpy.argmax(points.Y_SEQUENCE, -1).astype("float32")
    y = y.reshape(y.shape[0] * y.shape[1])
    dset3 = f.create_dataset('actual_ids', y.shape, dtype='int')
    dset3[...] = y

    logger.debug("P shape: %s, X shape: %s, P shape: %s,  Y shape: %s" %
                 (str(states.shape), str(X_flat.shape), str(p.shape),
                  str(y.shape)))

    with open(join(outpath, "words.dict"), mode='w') as f:
        for word, idx in index.word_index.items():
            if len(word) == 0:
                word = "_"
            word = word.replace(" ", "_")
            f.write("%s %s\n" % (word, idx))

#     with open("entities.dict", mode="w") as f:
#         for entity, idx in index.entity_index.items():
#             f.write("%s %s\n" % (entity, idx))

    with open(join(outpath, "labels.dict"), mode="w") as f:
        for label, idx in index.class_index.items():
            f.write("%s %s\n" % (label, idx))

    generateYAML(model, outpath)


def path_to_str(i, X, index):
    path = X[i]
    
    return " ".join([index.getWord(x) for x in path[numpy.where(path != 0)[0]]])


def toManifoldCluster(index: Index, X, X_path, Y, N_classes, outpath, label):    
    '''
    Creates a 2-dimensional representation of the data in X. Removes extreme outliers from the manifold. Clusters the points in the representation using K-Means. Writes a plot of the manifold and the clustering such that clustered regions are indicated by the same color. Manifold points are annotated with an ID. Writes a file which maps IDs to dependency paths.
    :param index:
    :param X:
    :param X_path:
    :param Y:
    :param N_classes:
    :param outpath:
    :param label:
    '''
    
    
    # create a 2-dimensional manifold
    logger.info("Computing the manifold embedding...")
    X = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    
    # remove outliers from the manifold
    outlier_classifier = EllipticEnvelope(contamination=.1)
    outlier_classifier.fit(X)
    inlier_indices = numpy.where(outlier_classifier.predict(X) == 1)[0]
    
    # we need this to see how much points we removed from the original data
    old_lenx = len(X)
    
    # select only inliers from our data
    X = X[inlier_indices]
    X_path = X_path[inlier_indices]
    Y = Y[inlier_indices]
        
    # train a clustering model
    logger.debug("Clustering...")
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.15)
    
    if bandwidth == 0:
        logger.warn("Bandwith is 0! Setting bandwith to 1")
        bandwidth = 1
    
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
#     kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
#     kmeans.fit(X)
    
    # now, we create the plot and the accompanying text file
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.05  # point in the mesh [x_min, x_max]x[y_min, y_max].
#     
#     # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1 , X[:, 0].max() + 1 
    y_min, y_max = X[:, 1].min() - 1 , X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    
    logger.debug("Predicting for %s\t%s" % (str(xx.shape), str(yy.shape)))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = ms.predict(numpy.c_[xx.ravel(), yy.ravel()])
    
    logger.debug("Rendering...")
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    
    lines = []
    
    # create the arrows with IDs pointing to the respective points
    for i in range(len(X)):
        x0 = X[i, 0]
        x1 = X[i, 1]
        plt.annotate(str(i), xy=(x0, x1), xytext=(15, 15), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        lines.append([i, path_to_str(i, X_path, index), (x0, x1)])
    
    # Plot the centroids as a white X
    centroids = ms.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    
    plt.title('MeanShift for %s, removed %d outliers\n' % (label.upper(), old_lenx - len(inlier_indices)))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 2048
    fig_size[1] = 1024
    plt.rcParams["figure.figsize"] = fig_size
    
    # save the figure
    plt.savefig(join(outpath, label + ".pdf"), format="pdf", dpi=960)
    
    # create a table with ID, (x0,x1), path and write it to file
    rows = [["id", "path", "(x0,x1)"]]
    rows += lines
    
    final_string = tabulate(rows, headers="firstrow")
    
    # write the table
    with open(join(outpath, label + ".txt"), "w") as f:
        f.write(final_string)

def toManifoldVis(index:Index, X, X_path, Y, N_classes, outpath):
    
    logger.debug("X shape: %s, Y shape: %s" % (str(X.shape), str(Y.shape)))
    plt.title("TSNE argument prediction visualization", fontsize=14)

    logger.debug("Reshaped X: %s" % str(X.shape))
    logger.info("Computing the manifold embedding...")

    X = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
        
    logger.debug("Manifold: %s" % (str(X.shape)))
    logger.info("Plotting...")
    X_ar = {}
    X_ar_indices = {}
     
    f, plots = plt.subplots(1, N_classes, sharey=True, figsize=(300, 15))
     
    N = len(X)
     
    lines = []
     
    for i in range(len(index.class_index)):
         
        lines.append(index.getClass(i))
         
        for x in range(N):
 
            if Y[x] == i:
                X_ar.setdefault(i, [])
                X_ar_indices.setdefault(i, [])
                X_ar[i].append(X[x])
                X_ar_indices[i].append(x)
                 
        if  i in X_ar:
             
            plots[i].plot([x[0] for x in X_ar[i]], [x[1] for x in X_ar[i]], "ro", label=index.getClass(i), color=plt.cm.Set1(i / 20.))
            plots[i].legend()
             
            already_assigned = []
             
            for j in range(len(X_ar[i])):
                 
                x0 = X_ar[i][j][0]
                x1 = X_ar[i][j][1]
                 
                if (x0, x1) not in already_assigned:
#                 
                    already_assigned.append((x0, x1))
                     
                    plots[i].annotate(str(j), xy=(x0, x1), xytext=(15, 15), textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
                     
                    lines.append("\t%d\t%s" % (j, str([index.getWord(x) for x in X_path[X_ar_indices[i][j]]])))
     
    plt.savefig(join(outpath, "plot.png"))
    with open(join(outpath, "indices_for_plot.txt"), "w") as f:
        f.write("\n".join(lines))
         
    print("Done.", flush=True)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, Y, index, title=None):
    x_min, x_max = numpy.min(X, 0), numpy.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1], "ro",
                 color=plt.cm.Set1(Y[i] / 10.), label=index.getArgument(Y[i]))
#                  fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def toSingleHTML(index: Index, points, MAX_LEN, final_output, sequence_output,
                 suffix, outpath):
    doc = dominate.document(title="visual")

    predictions, golds = final_output

#     if sequence_output is None or sequence_output is (None, None):
#         sequence_predictions = None
#         sequence_golds = None
#     else:
#         sequence_predictions, sequence_golds = sequence_output
#         logger.debug("Sequence predictions shape:%s" % str(sequence_predictions.shape))
#         logger.debug("Sequence golds shape:%s" % str(sequence_golds.shape))

    logger.debug("Predictions shape:%s" % str(predictions.shape))
    logger.debug("Golds shape:%s" % str(golds.shape))

    with doc:
        link(rel="stylesheet", href="style.css")
    
    with doc:
        for i in range(len(points.X_DYNAMIC)):
            sentence = numpy.zeros(len(points.X_DYNAMIC[i]))

            for j in range(len(points.X_DYNAMIC[i])):
                if points.X_DYNAMIC[i][j] != 0:
                    sentence[j] = points.X_DYNAMIC[i][j]
                else:
                    sentence[j] = points.X_STATIC[i][j]

            with div():

#                 if sequence_golds is not None:
#                     with div():
#                         attr(cls="sequence")
#                         for j in range(len(sequence_predictions[i])):
# 
#                             if sequence_golds[i][j] == 0:
#                                 continue
# 
#                             with div():
#                                 attr(cls="sequenceelement")
#                                 div(index.getClass(sequence_predictions[i][j]))
#                                 div(index.getClass(sequence_golds[i][j]))
#                                 div(index.getWord(sentence[j]))

                attr(cls="sentence")
                prediction = index.getClass(predictions[i])
                actual = index.getClass(golds[i])

                if prediction == actual:
                    cls = "tp"
                else:
                    cls = "fp"

                with div("is: " + prediction + ", should: " + actual):
                    attr(cls=cls)
                    pass

                div(index.getEntity(points.ENTITY[i]) + ", " + index.getEvent(points.EVENT[i]))

                for j in range(len(sentence)):
                    widx = sentence[j]
                    if widx == 0:
                        continue
                    word = index.getWord(widx)
                    span(word)
                br()

                for widx in points.CNN_LEFT[i]:

                    if widx == 0:
                        continue

                    word = index.getWord(widx)  #
                    span(word)

                br()

                for widx in points.CNN_RIGHT[i]:

                    if widx == 0:
                        continue

                    word = index.getWord(widx)  #
                    span(word)

                br()

                for widx in points.ENTIRE_SENTENCE[i]:

                    if widx == 0:
                        continue

                    word = index.getWord(widx)  #
                    span(word)

            hr()
    with open(join(outpath, "visual.{:s}.html".format(str(suffix))), "w") as file:
        file.write(str(doc))

    with open(join(outpath, "style.css"), "w") as file:
        file.write(".sequenceelement{display:inline-block}\n.sentence{}\n.wordblock{display:inline-block}\n.win{margin: 0 auto; color:black}\n.tp{color:green}\n.fp{color:red}\n.class_Label{font-size:0.75em}\n.trig{text-decoration: underline}\n.dep{color:blue}")




def manifoldPlaceTime(index, X, Y, N_classes, outpath):
    
    plt.close()
    
    logger.info("Plotting PLACE vs. TIME")

#     X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    plt.title("PLACE/TIME")
    place_indices = numpy.where(Y == index.getClassIndex("PLACE", False))[0]
    time_indices = numpy.where(Y == index.getClassIndex("TIME", False))[0]
    
    Y_place = numpy.zeros_like(Y)
    Y_place[place_indices] = 1
    
    Y_time = numpy.zeros_like(Y)
    Y_time[time_indices] = 1
    
    X_manifold = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    X_place = X_manifold[place_indices]
    X_time = X_manifold[time_indices]
    
#     X_place = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X[place_indices])
#     X_time = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X[time_indices])
    
#     assert not X_place != X_time

    plt.plot(X_time[:, 0], X_time[:, 1], "ro", X_place[:, 0], X_place[:, 1], "bo")
    
    plt.show()
    
#     plt.legend()
    plt.savefig(join(outpath, "place_vs_time.png"))


def performTSNEAnalysis(index, points, X, Y, N_classes, outpath):
    
    logger.debug("X %s Y %s" % (str(X.shape), str(Y.shape)))
    
    print("-------- 1 ---------")
    print("\n----------------\n".join([str(x) for x in X[:10, :, :10]]))
    print("-------- 2 ---------")
    print("\n----------------\n".join([str(x) for x in X[:10, :]]))
#     print("\n----------------\n".join([str([[index.getClass(z) for z in numpy.where(y == 1)[0]] for y in x]) for x in X[:10]]))
    print("-------- 3 ---------")
    print("\n----------------\n".join([str([index.getClass(z) for z in numpy.argmax(y, -1)]) for y in model.get_output(points)[1][:10, :]]))
    
    assert X.shape[0] == Y.shape[0]
    
    # points.LENGTH contains the length of a dependency path (= numbers of dependency edges). The index of the last element is always length*2.
    X = X[numpy.arange(len(X)), points.LENGTH.astype("int32") * 2]
        
    place_indices = numpy.where(Y == index.getClassIndex("PLACE", False))[0]
    time_indices = numpy.where(Y == index.getClassIndex("TIME", False))[0]
    
    X_TOTAL = points.X_STATIC + points.X_DYNAMIC
    
    toManifoldCluster(index, X[place_indices], X_TOTAL[place_indices], Y[place_indices], N_classes, outpath, "place")
    toManifoldCluster(index, X[time_indices], X_TOTAL[time_indices], Y[time_indices], N_classes, outpath, "time")
    
#     manifoldPlaceTime(index, X[place_indices], Y[place_indices], N_classes, outpath)


