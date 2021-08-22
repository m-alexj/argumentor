'''
Created on Dec 2, 2016

@author: judeaax
'''
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.core import Reshape, Masking, Flatten, Dense, Activation, \
    Dropout, RepeatVector, Lambda
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from model import logger
from keras.engine.training import Model
from basics.index import Index
from layers.attention_lstm import AttentionLSTM
from layers.masked import RemoveMask, MaskedSoftmax
from keras import backend as K
import numpy
from keras.layers.normalization import BatchNormalization
from keras.activations import tanh, hard_sigmoid
from dominate.tags import output
from layers.reverse import Reversed
from audioop import reverse
from keras.layers.merge import Multiply, Add, Average, Concatenate
from keras.engine.topology import Input


def make_cnn(cnn_filters: int, region: int, cnn_input, dropout=0.3):
    '''
    Creates three CNNs, one of widths 2, 3, 4.
    CNNs use the tanh activation and are followed
    by max pooling.

    :param cnn_filters: Number of filters per CNN
    :param region: Max pooling region.
    :param cnn_input: Input tensor
    :param dropout: A dropout value
    '''
    cnns = []
    for length in [2, 3, 4]:
        cnn = Convolution1D(cnn_filters, length)(cnn_input)
        if dropout:
            cnn = Dropout(dropout)(cnn)
        cnn = Activation("tanh")(cnn)
        cnn = MaxPooling1D(region - length - 1)(cnn)
        cnn = Reshape((cnn_filters,))(cnn)
        cnns.append(cnn)
    return cnns


def create_model(MAX_LEN: int, MAX_MENTION_LEN: int,
                 MAX_SENTENCE: int, N: int, N_classes: int, N_pos: int,
                 index: Index, embedding_weights, bias_size=150, lstm_size=200,
                 cnn_filters=50, dropout=0.3):
    '''
    Creates a Keras model. The model consists mainly of a biLSTM processing
    dependency paths, CNNs processing sentence tokens (up to a max sentence
    length), and a representation of the tuple (event type, entity mention,
    genre), also called the bias representation. Finally, a linear layer
    connects everything and produces a probability distribution over all
    argument classes.

    A dependency path always goes from an event trigger to an argument
    candidate.

    :param MAX_LEN: Max sequence length
    :param MAX_MENTION_LEN: Max number of tokens in a mention
    :param MAX_SENTENCE: Max number of tokens in a sentence
    :param N: Dimensionality of embeddings
    :param N_classes: Number of argument classes
    :param N_pos: Dimensionality of position embeddings
    :param index: An index
    :param embedding_weights: Pre-trained embedding weights if dimensionality N
    :param bias_size: Dimensionality of the bias representation
    :param lstm_size: Dimensionality of LSTM hidden states
    :param cnn_filters: Number of CNN filters
    :param dropout: Dropout value
    '''

    # these are needed for the bias representation
    entity_dimensions = N
    event_dimensions = N
    genre_dimensions = N

    logger.debug("Bias size: %d, LSTM size: %d, CNN filters: %d, dropout: %f" %
                 (bias_size, lstm_size, cnn_filters, dropout))

    # === the inputs ===
    restrictions_input = Input(
        shape=(N_classes,), name="restr_input", dtype=K.floatx())

    entity_sequence_input = Input(
        shape=(MAX_LEN,), name="entity_sequence_input", dtype=K.floatx())

    sequence_restrictions_input = Input(
        shape=(MAX_LEN, N_classes,), name="sequence_restrictions_input",
        dtype=K.floatx())

    word_dyn_input = Input(shape=(MAX_LEN,), name="word_dyn_input")

    word_stat_input = Input(shape=(MAX_LEN,), name="word_stat_input")

    word_dyn_reverse_input = Input(
        shape=(MAX_LEN,), name="word_dyn_reverse_input")

    word_stat_reverse_input = Input(
        shape=(MAX_LEN,), name="word_stat_reverse_input")

    raw_input = Input(shape=(MAX_LEN,), name="raw_input")

    word_trigger_position_input = Input(
        shape=(MAX_LEN,), name="word_trigger_position_input")

    word_mention_position_input = Input(
        shape=(MAX_LEN,), name="word_mention_position_input")

    word_trigger_position_reverse_input = Input(
        shape=(MAX_LEN,), name="word_trigger_position_reverse_input")

    word_mention_position_reverse_input = Input(
        shape=(MAX_LEN,), name="word_mention_position_reverse_input")

    word_mask = Input(shape=(MAX_LEN,), name="word_mask")

    pos_input = Input(shape=(MAX_LEN,), name="pos_input")

    genre_input = Input(shape=(1,), name="genre_input")

    mention_token_input = Input(shape=(1,), name="mention_token_input")

    trigger_token_input = Input(shape=(1,), name="trigger_token_input")

    entity_input = Input(shape=(1,), name="entity_input")

    entity_sub_input = Input(shape=(1,), name="entity_sub_input")

    mention_input = Input(shape=(1,), name="mention_input")

    event_input = Input(shape=(1,), name="event_input")

    common_ancestor_input = Input(shape=(1,), name="common_ancestor_input")

    # all the embeddings we use
    word_dyn_embeddings = Embedding(len(index.word_index), N, trainable=True,
                                    weights=[embedding_weights])
    word_stat_embeddings = Embedding(len(index.word_index), N, trainable=False,
                                     weights=[embedding_weights])
    embedded_positions = Embedding(len(index.position_index), N_pos,
                                   trainable=True)
    EntityEmbeddings = Embedding(len(index.entity_index), entity_dimensions)
    embedded_entity_type = EntityEmbeddings(entity_input)
    embedded_entity_subtype = EntityEmbeddings(entity_sub_input)
    embedded_mention_type = EntityEmbeddings(mention_input)
    embedded_event_type = Embedding(len(index.event_index),
                                    event_dimensions)(event_input)
    embedded_genre = Embedding(len(index.genre_index),
                               genre_dimensions)(genre_input)

    # the bias representation
    initial_bias_n = 3 * entity_dimensions + event_dimensions + genre_dimensions
    bias = Reshape((initial_bias_n,))(Concatenate()([embedded_entity_type,
                                                     embedded_entity_subtype,
                                                     embedded_mention_type,
                                                     embedded_event_type,
                                                     embedded_genre]))
    bias = Dense(int(initial_bias_n / 2), activation="relu")(bias)
    logger.debug("Intermediate bias repr: %s" % (str(bias)))
    bias = Dense(bias_size, activation="relu")(bias)
    logger.debug("Bias vector: %s" % str(bias))

    # the sentence CNNs
    sentence_input = Input(shape=(MAX_SENTENCE,), name="sentence_input")
    sentence_mask_input = Input(shape=(MAX_SENTENCE,), name="sentence_mask_input")
    sentence_trigger_position_input = Input(shape=(MAX_SENTENCE,),
                                            name="sentence_trigger_position_input")
    sentence_mention_position_input = Input(shape=(MAX_SENTENCE,),
                                            name="sentence_mention_position_input")
    sentence = Concatenate()([RepeatVector(MAX_SENTENCE)(bias), word_stat_embeddings(sentence_input), embedded_positions(sentence_trigger_position_input), embedded_positions(sentence_mention_position_input)])
    reshaped_sentence_mask = Reshape((MAX_SENTENCE, 1))(sentence_mask_input)
    reshaped_sentence_mask = Lambda(lambda x: K.repeat_elements(x, N + 2 * N_pos + bias_size, 2), output_shape=(MAX_SENTENCE, N + 2 * N_pos + bias_size))(reshaped_sentence_mask)
    sentence = Multiply()([sentence, reshaped_sentence_mask])
    if cnn_filters > 0:
        cnns = []
        cnns += make_cnn(cnn_filters, MAX_SENTENCE, sentence, dropout=0.0)

    # the dependency path biLSTM
    embedded_dependency_path = Add()([word_dyn_embeddings(word_dyn_input), word_stat_embeddings(word_stat_input)])

    reshaped_to_maxlen_word_mask = Reshape((MAX_LEN, 1))(word_mask)
    reshaped_word_mask = Lambda(lambda x: K.repeat_elements(
        x, N + bias_size + 2 * N_pos, 2), output_shape=(
            MAX_LEN, N + bias_size + 2 * N_pos))(reshaped_to_maxlen_word_mask)
    lexical_depenency_embeddings = Concatenate()([
        embedded_dependency_path, embedded_positions(
            word_trigger_position_input),
        embedded_positions(word_mention_position_input),
        RepeatVector(MAX_LEN)(bias)])
    lexical_depenency_embeddings = Multiply()([lexical_depenency_embeddings,
                                               reshaped_word_mask])
    lexical_depenency_embeddings = Masking(mask_value=0.0)(
        lexical_depenency_embeddings)
    logger.debug("Lexical dependency embeddings: %s" % str(
        lexical_depenency_embeddings))

    lex_dep_blstm = LSTM(lstm_size, return_sequences=True, implementation=2)(
        lexical_depenency_embeddings)
    lex_dep_blstm = RemoveMask()(lex_dep_blstm)

    lex_dep_blstm_b = LSTM(lstm_size, go_backwards=True, return_sequences=True,
                           implementation=2)(lexical_depenency_embeddings)
    lex_dep_blstm_b = RemoveMask()(lex_dep_blstm_b)
    lex_dep_blstm_b = Reversed(1)(lex_dep_blstm_b)
    lex_dep_blstm_b = Multiply()([lex_dep_blstm_b, Lambda(lambda x: K.repeat_elements(x, lstm_size, 2), output_shape=(MAX_LEN, lstm_size))(reshaped_to_maxlen_word_mask)])

    # after computing a forward and a backward pass
    # of the path, we average them and apply a dropout
    path = Average()([lex_dep_blstm, lex_dep_blstm_b])
    ppath = path
    path = Reversed(1)(path)
    path = Dropout(dropout)(path)
    logger.debug("Path: %s" % str(path))

    # the final layer and the final softmax
    final_sequence = TimeDistributed(Dense(N_classes, activation="linear"))(
        path)
    sequence_restrictions = Reversed(1)(sequence_restrictions_input)
    final_sequence = Multiply()([final_sequence, sequence_restrictions])
    final_sequence = TimeDistributed(MaskedSoftmax())(final_sequence)

    reshaped_seq_out_mask = Reshape((MAX_LEN, 1))(word_mask)
    reshaped_seq_out_mask = Lambda(lambda x: K.repeat_elements(x, N_classes, 2), output_shape=(MAX_LEN, N_classes))(reshaped_seq_out_mask)
    final_sequence = Multiply(name="final_sequence")([final_sequence,
                                                      reshaped_seq_out_mask])

    final_sequence_argmax = Lambda(lambda x: K.cast(K.argmax(x, -1),
                                                    dtype=K.floatx()),
                                   output_shape=(MAX_LEN,))(final_sequence)

    final_merge = Concatenate(axis=-1)([Flatten()(
        path)] + cnns) if cnn_filters > 0 else Flatten()(path)
    final_merge = Dropout(dropout)(final_merge)
    logger.debug("Final merge: %s" % (str(final_merge)))
    final = Dense(N_classes, activation="linear")(final_merge)
    final = Multiply()([final, restrictions_input])
    # we catch this to return a model which outputs the final
    # layer before softmax; this is used to visualize TSNE representation
    # of event arguments
    # ppath = final
    final = MaskedSoftmax(name="final_out")(final)
    # Keras definition of inputs
    inputs = [entity_sequence_input, word_dyn_input, word_dyn_reverse_input, word_stat_input, word_stat_reverse_input, raw_input, word_trigger_position_input, word_trigger_position_reverse_input, word_mention_position_input, word_mention_position_reverse_input, word_mask, pos_input, genre_input, entity_input, entity_sub_input, mention_input, event_input, restrictions_input, sequence_restrictions_input, mention_token_input, trigger_token_input, sentence_input, sentence_trigger_position_input, sentence_mention_position_input, sentence_mask_input, common_ancestor_input]
    # Keras definition of the model
    model = Model(inputs=inputs, outputs=[final])
    final_sequence_argmax_model = Model(inputs=inputs, outputs=[final_sequence_argmax])
    # this version of the model outputs the final tensor before softmax
    forward_path_model = Model(inputs=inputs, outputs=[ppath])

    return model, final_sequence_argmax_model, forward_path_model


def get_input_indexed(points, indices):
    '''
    Returns 'sliced' input samples. From all input samples, returns only those
    specified by the given indices.
    :param points: Some points
    :param indices: Some indices
    '''
    return [points.ENTITY_SEQUENCE[indices], points.X_DYNAMIC[indices], points.X_DYNAMIC_REVERSE[indices], points.X_STATIC[indices], points.X_STATIC_REVERSE[indices], points.X_RAW[indices], points.P_trigger[indices], points.P_trigger_reverse[indices], points.P_argument[indices], points.P_argument_reverse[indices], points.X_MASK[indices], points.X_POS[indices], points.GENRE[indices], points.ENTITY[indices], points.ENTITY_SUB[indices], points.MENTION[indices], points.EVENT[indices], points.R[indices], points.R_SEQUENCE[indices], points.MENTION_TOKEN[indices], points.TRIGGER_TOKENS[indices], points.SENTENCE[indices], points.SENTENCE_TRIGGER_POSITIONS[indices], points.SENTENCE_MENTION_POSITIONS[indices], points.SENTENCE_MASK[indices], points.COMMON_ANCESTOR[indices]]


def get_input(points):
    '''
    Returns input samples.
    :param points: Some points
    '''
    return get_input_indexed(points, slice(None))


def get_output(points):
    return get_output_indexed(points, slice(None))


def get_output_indexed(points, indices):
    return [points.Y[indices], numpy.flip(points.Y_SEQUENCE[indices], 1)][0]


def get_weights(points):
    return get_weights_indexed(points, slice(None))


def get_weights_indexed(points, indices):
    return [points.W[indices], points.W[indices]][0]


def get_argmax_input(points):
    return get_input(points)


def get_argmax_output(points):
    outp = get_output(points)
    if isinstance(outp, list) and len(outp) > 1:
        outp = outp[1]
    argm = numpy.argmax(outp, -1)
    print(argm.shape, outp.shape)
    return argm

