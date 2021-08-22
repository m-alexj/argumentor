'''
Created on Oct 21, 2016

@author: judeaax
'''
import numpy
import logging
import sys

class Index(object):
    '''
    classdocs
    '''

    UNKNOWN = "UNKNOWN"
    NULL = "NULL"
    class_index = {}
    inv_class_index = {}
    
    word_index = {}
    inv_word_index = {}
    
    position_index = {}
    inv_position_index = {}
    
    entity_index = {}
    inv_entity_index = {}

    event_index = {}
    inv_event_index = {}

    genre_index = {}
    inv_genre_index = {}

    class_restrictions = {}

    logger = logging.getLogger(__name__)


    def getEntity(self, idx):
        return self.inv_entity_index[idx]

    def getEvent(self, idx):
        return self.inv_event_index[idx]
    
    def getEntityIndex(self, entity, add):
        return self.resolve(entity, self.entity_index, self.inv_entity_index, add)
    
    
    def getEventIndex(self, event, add):
        return self.resolve(event, self.event_index, self.inv_event_index, add)
    
    
    def __init__(self, w2v):
        self.w2v = w2v
        self.getClassIndex(self.UNKNOWN, True)
        # we have to make sure that "NULL" has index 1
        self.getClassIndex(self.NULL, True)
        self.getWordIndex(self.UNKNOWN, True)
        self.getPositionIndex(self.UNKNOWN, True)
        self.getEntityIndex(self.UNKNOWN, True)
        self.getEventIndex(self.UNKNOWN, True)
        self.getGenreIndex(self.UNKNOWN, True)
    
    def addClass(self, cls, add):
        self.resolve(cls, self.class_index, self.inv_class_index, add)
    
    def getPosition(self, idx):
        return self.inv_position_index[idx]
    
    def addPosition(self, pos, add):
        self.resolve(pos, self.position_index, self.inv_position_index, add)
    
    def getPositionIndex(self, pos, add):
        return self.resolve(pos, self.position_index, self.inv_position_index, add)
    
    def getGenre(self, idx):
        return self.inv_genre_index[idx]
    
    def addGenre(self, pos, add):
        self.resolve(pos, self.genre_index, self.inv_genre_index, add)
    
    def getGenreIndex(self, pos, add):
        return self.resolve(pos, self.genre_index, self.inv_genre_index, add)
    
    def getClassIndex(self, cls, add):
        return self.resolve(cls, self.class_index, self.inv_class_index, add)
             
    
    def getClass(self, idx):
        if idx in self.inv_class_index:
            return self.inv_class_index[idx]
        else:
            return "UNKNOWN CLASS"
    
    def getWord(self, idx):
        return self.inv_word_index[idx]
    
    def getWordIndex(self, word, add, lowercase=True):
        
        if lowercase:
            values = [x.lower() for x in word.split("@")]
        else:
            values = [x for x in word.split("@")]
        if len(values) == 3:
            if values[2] + "@" + values[1] in self.w2v:
                word = values[2] + "@" + values[1]
            elif values[0] + "@" + values[1] in self.w2v:
                word = values[0] + "@" + values[1]
            else:
                if values[2] in self.w2v:
                    word = values[2]
                else:
                    word = values[0]
        else:
            word = values[0]

        idx = self.resolve(word, self.word_index, self.inv_word_index, add)
        
#         if idx == self.word_index[self.UNKNOWN]:
#             assert word == self.UNKNOWN or not add, word
#             return self.resolve("OOV", self.word_index, self.inv_word_index, True)
#         else:
        return idx
    
    def resolve(self, val, index, inv_index, add):
        if val not in index:
            
            if not add:
                if self.UNKNOWN in index:
                    return index[self.UNKNOWN]
                else:
                    print("%s" % val, file=sys.stderr, flush=True)
            
            index[val] = len(index)
            inv_index[index[val]] = val
        return index[val]
    
    def makeEmbeddingWeights(self, N):
        symbols = len(self.word_index)
        embedding_weights = numpy.zeros((symbols, N))
        
        misses = 0
        
        for word, index in self.word_index.items():
            
            if index == 0:
                continue
            
            if "->" not in word and "<-" not in word:
                if word.startswith(":"):
                    word = word[1:]
                elif "," in word and len(word) > 1:
                    word = word.replace(",", ".")
                
            if word in self.w2v:
                embedding_weights[index] = self.w2v[word]
            else:
                if "->" not in word and "<-" not in word:
                    
                    capital_word = " ".join(w.capitalize() for w in word.split())
                    
                    if capital_word in self.w2v:
                        embedding_weights[index] = self.w2v[capital_word]
                    else:
                        uppercase_word = word.upper()
                        if uppercase_word in self.w2v:
                            embedding_weights[index] = self.w2v[uppercase_word]
                        else:
                            self.logger.debug("Embedding miss: %s" % word)
                            misses += 1
                embedding_weights[index] = numpy.zeros(N)  # numpy.random.uniform(-0.01, 0.01, N)               
        return embedding_weights, misses
    
    def resolveWords(self, array):
        
        return [self.getWord(x) for x in array]
    
    def getClassRestrictions(self, ev_idx, en_idx, N_classes):
        
        if en_idx == self.getEntityIndex("dependency", False):
            d = numpy.zeros(N_classes)
            d[self.getClassIndex("dependency", False)] = 1
            return d
        
        if en_idx == self.getEntityIndex("TRIGGER", False):
            d = numpy.zeros(N_classes)
            d[self.getClassIndex("TRIGGER", False)] = 1
            return d
        
        if ev_idx not in self.class_restrictions or en_idx not in self.class_restrictions[ev_idx]:
            d = numpy.zeros(N_classes)
            d[self.getClassIndex("NULL", False)] = 1
            return d
        else:
            return self.class_restrictions[ev_idx][en_idx]
    
    def addClassRestriction(self, l_idx, ev_idx, en_idx:int, N_classes):
        
        if ev_idx not in self.class_restrictions:
            r = numpy.zeros(N_classes)
            r[l_idx] = 1
            r[self.getClassIndex("NULL", False)] = 1
            self.class_restrictions[ev_idx] = {en_idx: r}
        elif en_idx not in self.class_restrictions[ev_idx]:
            r = numpy.zeros(N_classes)
            r[l_idx] = 1
            r[self.getClassIndex("NULL", False)] = 1
            self.class_restrictions[ev_idx][en_idx] = r
        else:
            self.class_restrictions[ev_idx][en_idx][l_idx] = 1        
            
