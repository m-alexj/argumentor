'''
Created on Nov 23, 2016

@author: judeaax
'''
import logging
import numpy
from basics.points import Points
from keras.utils.np_utils import to_categorical
from basics.index import Index


logger = logging.getLogger("Argumentor.loader")


def loadData(file, MAX_LEN):
    logger.debug("Loading data from %s" % file)

    data = []
    sentence_counter = 1
    pair_counter = 1
    with open(file) as f:
        lines = f.read().splitlines()
        pair = [[sentence_counter, pair_counter]]
        for i in range(len(lines)):
            line = lines[i]

            values = line.split("\t")

#             if values[0].lower() in ["crime", "position", "money","pair","crime", "price"]:
#                 continue

#             if values[0] != 'NULL':
#                     values[0] = 'NON-NULL'
            if len(values) > 1:
                pair.append(values)
            else:
                if len(pair) > 1:
                    data.append(pair)
                    pair_counter += 1
                pair = [[sentence_counter, pair_counter]]
                
                if i < len(lines) - 1 and len(lines[i + 1].split("\t")) <= 1:
                    sentence_counter += 1
    
    logger.debug("Read %d samples" % len(data))
    logger.debug("Non-null samples: %d" % numpy.count_nonzero(numpy.array([x[1][0] != "NULL" for x in data], dtype="int8")))
    logger.debug("Read %d sentences" % sentence_counter)
    
    return data
            

def process_meta_data(i:int, line:[], index:Index, points:Points, N_classes, MAX_CNN_LEN, MAX_MENTION_LEN, addToIndex, training):
    label = line[0]
                                
    event_type = line[1]
    entity_type = line[2]
    entity_subtype = line[3]
    mention_type = line[4]
    ev_idx = index.getEventIndex(event_type, addToIndex)
    en_idx = index.getEntityIndex(entity_type, addToIndex)
    es_idx = index.getEntityIndex(entity_subtype, addToIndex)
    mt_idx = index.getEntityIndex(mention_type, addToIndex)
    
#     if training and index.getClassRestrictions(ev_idx, en_idx, N_classes)[index.getClassIndex(label, False)] == 0:
#         return 1
    
    points.ENTITY[i] = numpy.array(en_idx, dtype="int32")  # @NoEffect
    points.ENTITY_SUB[i] = numpy.array(es_idx, dtype="int32")  # @NoEffect
    points.MENTION[i] = numpy.array(mt_idx, dtype="int32")  # @NoEffect
    points.EVENT[i] = numpy.array(ev_idx, dtype="int32")
    points.Y[i] = to_categorical([index.getClassIndex(label, addToIndex)], N_classes)
    
    null_label = label == "NULL"
    
    if null_label:
        points.Y_BINARY[i] = numpy.array([0, 1, 0])
    else:
        points.Y_BINARY[i] = numpy.array([0, 0, 1])
    
    points.R[i] = index.getClassRestrictions(ev_idx, en_idx, N_classes)

    cnn_sequence = line[5].split("|||")
    
    mstart = -1
    mend = -1
    tstart = -1
    for x in range(len(cnn_sequence)):
        
        sl = cnn_sequence[x].split(":::")
        
        if x < len(points.ENTIRE_SENTENCE[i]):
            points.ENTIRE_SENTENCE[i][x] = index.getWordIndex(sl[0], True)
        
        if "_" in sl[2]:
            if sl[2] == "0_0":
                mstart = x
            else:
                mend = x
        if sl[1] == "0":
            tstart = 0
    
    if mend < 0:
        mend = mstart
    
    assert mstart >= 0 and mend >= 0 and tstart >= 0, "%d %d %d, line %d,\n%s" % (mstart, mend, tstart, i, str(line))
    
    writex = 0
    
    for x in range(len(cnn_sequence)):
        
        if x < mstart - 5 or x > mend + 5:
            continue
        
        sentence_word = index.getWordIndex(sl[0], True)
        
        sl = cnn_sequence[x].split(":::")
        points.SENTENCE[i][writex] = sentence_word
        points.SENTENCE_MASK[i][writex] = 1
        points.SENTENCE_TRIGGER_POSITIONS[i][writex] = index.getPositionIndex("stp %s" % sl[1], addToIndex)
        points.SENTENCE_MENTION_POSITIONS[i][writex] = index.getPositionIndex("smp %s" % sl[2], addToIndex)
        writex += 1
        
    genre = line[6]
    
    points.GENRE[i] = numpy.array(index.getGenreIndex(genre, addToIndex), dtype="int32")
    
    left_context = line[7].split(":::")[-MAX_CNN_LEN:]
    right_context = line[8].split(":::")[:MAX_CNN_LEN]
    
    left_values = numpy.array([index.getWordIndex(x, True) for x in left_context])
    left_mask = numpy.array([1 for x in left_context])
    
    if len(left_values) < MAX_CNN_LEN:
        left_values = numpy.pad(left_values, (MAX_CNN_LEN - len(left_values), 0), mode="constant")
        left_mask = numpy.pad(left_mask, (MAX_CNN_LEN - len(left_mask), 0), mode="constant")
    
    left_positions = numpy.zeros_like(left_values)
    
    for l in range(len(left_values)):
        if left_values[l] != 0:
            left_positions[l] = index.getPositionIndex(l - MAX_CNN_LEN, addToIndex)
    
    points.CNN_LEFT[i] = left_values
    points.CNN_LEFT_MASK[i] = left_mask
    points.CNN_LEFT_POSITIONS[i] = left_positions
        
    right_values = numpy.array([index.getWordIndex(x, True) for x in right_context])
    right_mask = numpy.array([1 for x in right_context])
    
    if len(right_values) < MAX_CNN_LEN:
        right_values = numpy.pad(right_values, (0, MAX_CNN_LEN - len(right_values)), mode="constant")
        right_mask = numpy.pad(right_mask, (0, MAX_CNN_LEN - len(right_mask)), mode="constant")
    
    right_positions = numpy.zeros_like(right_values)
    
    for l in range(len(right_values)):
        if right_values[l] != 0:
            right_positions[l] = index.getPositionIndex(l + 1, addToIndex)
    
#                 print([index.getPosition(x) for x in right_positions], [index.getWord(x) for x in right_values], right_context, right_mask)
#                 print([index.getPosition(x) for x in left_positions], [index.getWord(x) for x in left_values], left_context, left_mask)
    
    points.CNN_RIGHT[i] = right_values
    points.CNN_RIGHT_MASK[i] = right_mask
    points.CNN_RIGHT_POSITIONS[i] = right_positions
    mention_tokens = line[9].split(":::")
    
#     mention_token_idx = index.getWordIndex(mention_tokens[-1], True)
#     points.MENTION_TOKEN[i] = mention_token_idx
    
    trigger_tokens = line[8].split(":::")
    points.TRIGGER_TOKENS[i] = index.getWordIndex(trigger_tokens[0], True)
    
    return 0, null_label, ev_idx, en_idx


def process_path_element(label, word, pos, ev_idx, en_idx, i:int, write_index:int, reverse_write_index, index:Index, position_trigger, position_argument, points:Points, dependency_directions, N_classes, addToIndex):
    
    points.Y_SEQUENCE[i][write_index] = to_categorical([index.getClassIndex(label, addToIndex)], N_classes)
    points.R_SEQUENCE[i][write_index] = index.getClassRestrictions(ev_idx, en_idx, N_classes)        
    points.ENTITY_SEQUENCE[i][write_index] = en_idx
#     points.X[i, write_index] = index.getWordIndex(word, True,lowercase=False) 
    
    
    if "<-" in word or "->" in word or "no_path" == word:
        points.X_DYNAMIC[i][write_index] = index.getWordIndex(word, True)
        points.X_DYNAMIC_REVERSE[i][reverse_write_index] = index.getWordIndex(word, True)
        raw_ids = numpy.nonzero(points.X_RAW[i])[0]
        if len(raw_ids) == 0:
            raw_idx = -(points.LENGTH[i])
        else:
            raw_idx = raw_ids[-1] + 1  
#         print(raw_idx)
        points.X_RAW[i][raw_idx] = index.getWordIndex(word, True, lowercase=False)
#         print([index.getWord(x) for x in points.X_RAW[i]])
#         print(points.LENGTH[i])
    else:
        
        points.X_STATIC[i][write_index] = index.getWordIndex(word, True)
        points.X_STATIC_REVERSE[i][reverse_write_index] = index.getWordIndex(word, True)
    
    points.X_MASK[i, write_index] = 1
    if points.X_DYNAMIC[i, write_index] != 0 or points.X_STATIC[i, write_index] != 0:
        points.P_trigger[i, write_index] = index.getPositionIndex(position_trigger, addToIndex)
        points.P_argument[i, write_index] = index.getPositionIndex(position_argument, addToIndex)
        points.P_trigger_reverse[i][reverse_write_index] = index.getPositionIndex(position_trigger, addToIndex)
        points.P_argument_reverse[i][reverse_write_index] = index.getPositionIndex(position_argument, addToIndex)

    points.X_POS[i, write_index] = index.getWordIndex(pos, True, lowercase=False)


def structureData(data, MAX_LEN, MAX_CNN_LEN, MAX_MENTION_LEN, MAX_SENTENCE, N_classes, index, addToIndex, nonNullWeight=1, maxToLoad=-1, training=False):
    
    if maxToLoad != -1:
        data = data[:maxToLoad]
    
    points = Points()
    
    N = len(data)
    
    points.X_MASK = numpy.zeros((N, MAX_LEN))
#     points.X = numpy.zeros((N, MAX_LEN))
    points.X_STATIC = numpy.zeros((N, MAX_LEN))
    points.X_DYNAMIC = numpy.zeros((N, MAX_LEN))
    points.X_STATIC_REVERSE = numpy.zeros((N, MAX_LEN))
    points.X_DYNAMIC_REVERSE = numpy.zeros((N, MAX_LEN))
    points.X_POS = numpy.zeros((N, MAX_LEN))
    points.X_RAW = numpy.zeros((N, MAX_LEN))
    points.P_trigger = numpy.zeros((N, MAX_LEN))
    points.P_argument = numpy.zeros((N, MAX_LEN))
    points.P_trigger_reverse = numpy.zeros_like(points.P_trigger)
    points.P_argument_reverse = numpy.zeros_like(points.P_argument)
    points.Y_SEQUENCE = numpy.zeros((N, MAX_LEN, N_classes))
    points.Y = numpy.zeros((N, N_classes))
    points.Y_BINARY = numpy.zeros((N, 3))
    points.R = numpy.zeros_like(points.Y)
    points.R_SEQUENCE = numpy.zeros_like(points.Y_SEQUENCE)
    points.ENTITY_SEQUENCE = numpy.zeros_like(points.X_STATIC)
    points.W = numpy.zeros(N)
    
    points.ENTIRE_SENTENCE = numpy.zeros((N, 100))
    
    points.SENTENCE = numpy.zeros((N, MAX_SENTENCE))
    points.SENTENCE_MASK = numpy.zeros((N, MAX_SENTENCE))
    points.SENTENCE_TRIGGER_POSITIONS = numpy.zeros((N, MAX_SENTENCE))
    points.SENTENCE_MENTION_POSITIONS = numpy.zeros((N, MAX_SENTENCE))
    points.ENTITY = numpy.zeros((N))
    points.ENTITY_SUB = numpy.zeros((N))
    points.MENTION = numpy.zeros((N))
    points.EVENT = numpy.zeros((N))
    
    points.GENRE = numpy.zeros((N))
    
    points.TRIGGER_TOKENS = numpy.zeros((N))
    points.MENTION_TOKEN = numpy.zeros((N))
#     points.MENTION_MASK = numpy.zeros((N, MAX_MENTION_LEN))
#     points.MENTION_POSITIONS = numpy.zeros((N, MAX_MENTION_LEN))
    
    points.CNN_LEFT = numpy.zeros((N, MAX_CNN_LEN))
    points.CNN_LEFT_POSITIONS = numpy.zeros((N, MAX_CNN_LEN))
    points.CNN_LEFT_MASK = numpy.zeros((N, MAX_CNN_LEN))
    
    points.CNN_RIGHT = numpy.zeros((N, MAX_CNN_LEN))
    points.CNN_RIGHT_POSITIONS = numpy.zeros((N, MAX_CNN_LEN))
    points.CNN_RIGHT_MASK = numpy.zeros((N, MAX_CNN_LEN))
    
    points.CNN_TOTAL = numpy.zeros((N, 2 * MAX_CNN_LEN + MAX_MENTION_LEN))
    points.CNN_TOTAL_POSITIONS = numpy.zeros((N, 2 * MAX_CNN_LEN + MAX_MENTION_LEN))
    points.CNN_TOTAL_MASK = numpy.zeros((N, 2 * MAX_CNN_LEN + MAX_MENTION_LEN))
    
    points.LENGTH = numpy.zeros(N, "int32")
    
    points.COMMON_ANCESTOR = numpy.zeros(N)
    
    points.SENTENCE_ID = numpy.zeros((N, 2), dtype="int32")
    
    points.DOCUMENT_ID = []
    
    points.TRIGGER_ID = []
    
    non_null_counter = 0
    
    skipped_number = 0
    
    for i in range(len(data)):
#         print("\n".join([str(x) for x in data[i]]))
        dependency_directions = set()
        
        write_index = 0
        
        for j in range(1, len(data[i])):
            if int((len(data[i]) - 2) / 2) <= int((MAX_LEN - 3) / 2):
                word = data[i][j][1].split("@")[0]
                if "<-" in word or "->" in word or "NO_PATH" == word:
                    points.LENGTH[i] += 1
                    dependency_directions.add(word[-2:])
            else:
                break

        
        for j in range(0, len(data[i])):
            
            line = data[i][j]

            if j == 0:
                points.SENTENCE_ID[i] = numpy.array(line) - 1
            elif j == 1:
                skipped, null_label, ev_idx, _ = process_meta_data(i, line, index, points, N_classes, MAX_CNN_LEN, MAX_MENTION_LEN, addToIndex, training)
                if skipped > 0:
                    skipped_number += skipped
                    break
                points.DOCUMENT_ID.append([line[9], data[i][0][1] - 1])
                points.TRIGGER_ID.append([line[9] + ":" + line[10], data[i][0][1] - 1])
            else:
                if int((len(data[i]) - 2) / 2) <= int((MAX_LEN - 3) / 2):
                    
                    label = line[0];
                    word = line[1].lower()                
                    position_trigger = "pt:" + line[2]
                    position_argument = "pa" + line[3]
                    pos = line[4]
                    en_idx = index.getEntityIndex(line[5], addToIndex)
                    if pos == "dependency":
                        pos = word
                        label = "dependency"
                        position_trigger = "dependency"
                        position_argument = "dependency"
                        en_idx = index.getEntityIndex("dependency", addToIndex)
                    
                    if j == 2:
                        label = "TRIGGER"
                        en_idx = index.getEntityIndex("TRIGGER", addToIndex)
                    
                    process_path_element(label, word, pos, ev_idx, en_idx, i, -(points.LENGTH[i] * 2 + 1) + write_index, -write_index - 1, index, position_trigger, position_argument, points, dependency_directions, N_classes, addToIndex)
                    write_index += 1
#         process_path_element("</s>", "</s>", "</s>", -1, -1, i, write_index, index, position_trigger, position_argument, points, dependency_directions, N_classes, addToIndex)
        if len(dependency_directions) > 1:
            points.COMMON_ANCESTOR[i] = index.getWordIndex("COMMON_ACESTOR:1", True, lowercase=False)
        else:
            points.COMMON_ANCESTOR[i] = index.getWordIndex("COMMON_ACESTOR:0", True, lowercase=False)
        
        points.MENTION_TOKEN[i] = points.X_STATIC[i, write_index - 1]
        
        assert points.LENGTH[i] * 2 <= MAX_LEN, "%d" % points.LENGTH[i]
        
#         print([index.getWord(x) for x in points.X_STATIC[i] + points.X_DYNAMIC[i]])
#         print([[index.getClass(y) for y in numpy.where(x==1.0)[0]] for x in points.R_SEQUENCE[i]])
#         print([index.getWord(x) for x in points.X_RAW[i]])
#         print(points.LENGTH[i])
#         print([x for x in points.X_MASK[i]])
#         print([index.getClass(numpy.argmax(x)) for x in points.Y_SEQUENCE[i]])
#         print([index.getWord(x) for x in points.X_STATIC_REVERSE[i] + points.X_DYNAMIC_REVERSE[i]])
#         print("---------------------------------")
#         print([x for x in points.X_MASK[i]])
        
        if null_label:
            points.W[i] = 1
        else:            
#             if index.getGenre(points.GENRE[i]).lower() in ['nw', 'bn', 'bc']:
#                 points.W[i] = nonNullWeight + 1
#             else:
            points.W[i] = nonNullWeight
            non_null_counter += 1
    
    logger.debug("Total support: {:d}".format(non_null_counter))
    for length in [1, 2, 3, 4, 5]:
        length_indices = numpy.where(points.LENGTH == length)[0]
        
        counter = 0
        for i in range(len(length_indices)):
            idx = length_indices[i]
            if numpy.argmax(points.Y_SEQUENCE[idx, -1]) != index.getClassIndex("NULL", False):
                counter += 1 
        
        logger.debug("Support for length {:d} paths: {:d}".format(length, counter))
    
    if training:
        logger.info("Skipped %d points because the restriction mask demanded it." % skipped)
    
    return points


def buildEventRestrictionsFor(N_classes, index: Index, data):
    
    counter = {}
    
    for i in range(len(data)):
        
        event_type = data[i][1][1]
        
        for j in range(2,len(data[i])):
            line = data[i][j]
    
            label = line[0]
            
            if j > 2:
                entity_type = line[5]
            elif j == 2:
                entity_type = "TRIGGER"
                    
                    
            label_idx = index.getClassIndex(label, True)
            event_idx = index.getEventIndex(event_type, True)
            entity_idx = index.getEntityIndex(entity_type, True)
            tuple_key = (label_idx, event_idx, entity_idx)
            counter.setdefault(tuple_key, 0)
            counter[tuple_key] += 1
            
            if ((label == "VICTIM" and (event_type in ["Conflict.Attack", "Movement.Transport"])) or (label == "PERSON" and event_type == "Life.Die") or (label == "AGENT" and event_type == "Conflict.Attack")):
                continue
            else :
                index.addClassRestriction(label_idx, event_idx, entity_idx, N_classes)


def buildEventRestrictions(N_classes, index, *args):
    for arg in args:
        buildEventRestrictionsFor(N_classes, index, arg)
