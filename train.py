#import inp
import itertools
import json
import os
import numpy as np
import math
from keras.preprocessing.sequence import pad_sequences
from model import GenerationModel, save_model, load_model, GenerationCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model as load_model_
import keras.backend as K
import sentencepiece as spm
from sampler import stats2intro_sampler as sampler
import collections

# Config GPU memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


ID,FORM,LEMMA,FEATS,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)
def build_vocabularies(documents):
    char_vocab={"<PADDING>":0,"<OOV>":1,"<BOS>":2,"<EOS>":3,"<BOD>":4,"<EOD>":5,"<BOW>":6,"<EOW>":7}
    for document,meta in documents:
        for comment,sent in document:
            for cols in sent:
                for char in cols[FORM]:
                    char_vocab.setdefault(char,len(char_vocab))
    return char_vocab


def vectorize_doc(document,char_vocab):
    doc_char_vectorized=[]
    for comment,sent in document:
        sent_char_vectorized=[]
        for cols in sent:
            sent_char_vectorized.append(list(char_vocab.get(char,1) for char in cols[FORM]))
        doc_char_vectorized.append(sent_char_vectorized)
    return doc_char_vectorized


def infinite_data_vectorizer(vocabulary, data_file):
    pass


def infinite_datareader(fname, max_documents=0):
    document_counter=0
    iteration=0
    while True:
        iteration+=1
        print("Iteration:", iteration)
        for document, meta in inp.get_documents(fname):
            document_counter+=1
            yield document, iteration, document_counter-1
            if max_documents!=0 and document_counter>=max_documents:
                break

def infinite_vectorizer(vocabulary, fname, batch_size, sequence_len, sp_model=None):
    """ ... """

    contexts=np.zeros((batch_size, sequence_len)) # Context input to encoder
    inputs=np.zeros((batch_size, sequence_len)) # Ground-truth input to decoder, 0==<unk> (padding)
    outputs=np.zeros((batch_size, sequence_len))
    batch_i = 0
    th = 0.25
    for document, iteration, doc_id in infinite_datareader(fname):
        th = min(th+0.0000005, 0.25)
        if doc_id % 1000 == 0:
            print("Word dropout:", th)

        # Warning: doc_id is a counter that doesn't reset at next iteration
        # vectorize_doc: document is a list of sentences, sentence is a list of words
        #  ... and word is a list of characters
        # --> flatten this to get a document as a list of characters, and add an end-of-word
        #  ... markers to represent white space

        doc = []
        for comment,sent in document:
            sent_str = ' '.join([cols[FORM] for cols in sent])
            sent_str =  sent_str.replace(' .', '.')\
                                .replace(' ,', ',')\
                                .replace(' :', ':')
            doc.append(sent_str)
            #if len(doc) >= iteration+1:
            #    break # limit doc size to 2 sents

        #doc = doc[-2:]
        for d_i in range(iteration%2, len(doc)-1,2):
            #doc_str = ' '.join(doc)
            input_str = doc[d_i]+' '+doc[d_i+1]
            input_pcs = [sp_model.PieceToId('<s>')]+sp_model.EncodeAsIds(input_str)+[sp_model.PieceToId('</s>')]

            #th = np.random.random()*0.5*(1-doc_id%10000/10000)+0.2
            ##th = 1-doc_id/2000000
            ## Context word dropout
            drop_words = (lambda words: [w for w in words if np.random.random() > th and w > 2])

            ##print(" doc id: %d (%d pcs)    " % (doc_id, len(doc_pc_ids)), end="")

            while len(input_pcs)>sequence_len:
                context = drop_words(input_pcs[:sequence_len])
                if context:
                    contexts[batch_i,-len(context):] = context
                inputs[batch_i,:] = input_pcs[:sequence_len]
                outputs[batch_i,:] = input_pcs[1:sequence_len+1]
                input_pcs = input_pcs[sequence_len:]
            else:
                try:
                    inputs[batch_i,-len(input_pcs)+1:] = input_pcs[:-1]
                    #inputs[batch_i,:len(input_pcs)] = input_pcs # Trailing padding
                    context = drop_words(input_pcs)
                    if context:
                        contexts[batch_i,-len(context):] = context # Leading padding
                except ValueError:
                    if len(input_pcs) == 1: # Skip <s>-only sequence
                        continue
                    else:
                        raise
                # Context: NUL NUL  ... ...
                # Input:   <s> ... </s> NUL
                # Output:  ... </s> NUL NUL

                outputs[batch_i,-len(input_pcs)+1:] = input_pcs[1:]
                #outputs[batch_i,:len(input_pcs)-1] = input_pcs[1:]
                ##yield ([inputs[:batch_i+1,:], inputs[:batch_i+1,:]], np.expand_dims(outputs[:batch_i+1,:], -1))#np.expand_dims(outputs,-1))

            if batch_i == batch_size-1:
                yield ([contexts, inputs], np.expand_dims(outputs,-1))
                contexts=np.zeros((batch_size, sequence_len))
                inputs=np.zeros((batch_size, sequence_len))
                outputs=np.zeros((batch_size, sequence_len))
                batch_i = 0
            else:
                batch_i += 1


#if __name__=="__main__":
import argparse
parser = argparse.ArgumentParser(description='')
g=parser.add_argument_group("Reguired arguments")

g.add_argument('--embedding_size', type=int, default=100, help='Character embedding size')
g.add_argument('--recurrent_size', type=int, default=200, help='Recurrent layer size')
g.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
g.add_argument('--hidden_dim', type=int, default=100, help='Size of the hidden layer in timedistributed')
g.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

args = parser.parse_args()
#model = train(args)


#def train(args):

sp = spm.SentencePieceProcessor()
sp.Load("m3k.model")
sp_vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
inv_sp_vocab = {value:key for key, value in sp_vocab.items()}
vocab = collections.defaultdict(lambda: len(vocab))

batch_size=10
sequence_len=200#250#1601
#n_docs = 1063180
input_len = 35

#train_vectorizer=infinite_vectorizer(sp_vocab, DATA_PATH, batch_size, sequence_len, sp_model=sp)
train_vectorizer = sampler(sp, vocab, input_len=input_len, output_len=sequence_len)

examples = [next(train_vectorizer) for i in range(10)]
example_input = [np.concatenate([inp[0] for inp, outp in examples]), np.concatenate([inp[1] for inp, outp in examples])]
example_output = np.concatenate([outp for inp, outp in examples])

generation_model=GenerationModel(len(sp_vocab), input_len, sequence_len, args)

#from keras.utils import multi_gpu_model
#para_model=multi_gpu_model(generation_model.model, gpus=4)

generate_callback=GenerationCallback(sp_vocab, sequence_len, [generation_model, generation_model.encoder_model, generation_model.decoder_model], sp_model=sp, generator=train_vectorizer, val_input=example_input)
checkpointer = ModelCheckpoint(filepath='models/02b_nomask.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
#earlystopper = EarlyStopping(patience=5, verbose=1)
#learnreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0000001, verbose=1)


# steps_per_epoch: how many batcher before running callbacks
# epochs: how many steps to run, should be basically infinite number (until killed)
#para_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
#para_model.fit_generator(train_vectorizer, steps_per_epoch=2000, epochs=1000000, verbose=1, callbacks=[generate_callback, checkpointer], validation_data=(example_input,example_output))

generation_model.model.load_weights('models/02b_nomask.hdf5')
init_ep = 0
hist = generation_model.model.fit_generator(train_vectorizer, steps_per_epoch=2000, initial_epoch=init_ep, epochs=2000, verbose=1, callbacks=[generate_callback,checkpointer], validation_data=(example_input,example_output))

#return generation_model
