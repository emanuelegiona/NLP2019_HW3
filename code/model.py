import functools
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet as wn
from math import ceil
from sklearn.utils import shuffle


import utils as u
import corpus_handler as ch


# only evalutate a function once, returning a member holding the instance for future calls
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
# --- --- ---


class MFSTagger:
    """
    Performs WSD by returning the most frequent sense (MFS) for each and every word in the sentence.
    """

    def __init__(self):
        pass

    @staticmethod
    def predict_word(word):
        """
        Returns the MFS for the given word.
        :param word: Lemma associated to the word to disambiguate
        :return: MFS for that lemma
        """

        synsets = wn.synsets(word)

        # in case the lemma isn't found in WordNet
        if synsets is None or len(synsets) == 0:
            return word

        synset = synsets[0]                         # fetch MFS
        return u.wn_id_from_synset(synset)

    @staticmethod
    def predict_sentence(sentence):
        """
        Returns MFS for every word in the given sentence.
        :param sentence: List of lemmas to disambiguate
        :return: List of most frequent senses corresponding to the given sentence
        """

        sent = []
        for word in sentence:
            sent.append(MFSTagger.predict_word(word))

        return sent


class BasicTagger:
    """
    Performs WSD as a sequence tagging task rather than per-word classification.
    """

    def __init__(self, learning_rate, embedding_size, hidden_size, num_layers, input_size, output_size):
        """
        Initializes a multi-layer Bi-LSTM model for WSD as means of sequence tagging.

        :param learning_rate: Learning rate to be used in the optimization
        :param embedding_size: Embedding size for the input vocabulary
        :param hidden_size: Hidden size for the LSTM networks
        :param num_layers: Number of LSTM layers to stack
        :param input_size: Size of the input vocabulary (plain words)
        :param output_size: Size of the output vocabulary (fine-grained senses)
        """

        assert num_layers > 0, "Layers number must be at least 1"

        self.lr = learning_rate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # correctly initialize graph
        self.sentence
        self.labels
        self.seq_lengths

        self.keep_prob
        self.rec_keep_prob

        self.embeddings
        self.lookup
        self.biLSTM

        self.dense_fine
        self.loss_fine

        self.loss
        self.train

    @lazy_property
    def sentence(self):
        with tf.variable_scope("sentence"):
            sentence = tf.placeholder(tf.int32, name="sentence", shape=[None, None])
            return sentence

    @lazy_property
    def labels(self):
        with tf.variable_scope("labels"):
            labels = tf.placeholder(tf.int32, name="labels", shape=[None, None])
            return labels

    @lazy_property
    def seq_lengths(self):
        with tf.variable_scope("seq_lengths"):
            seq_lengths = tf.placeholder(tf.int32, name="seq_lengths", shape=[None])
            return seq_lengths

    @lazy_property
    def keep_prob(self):
        with tf.variable_scope("keep_prob"):
            keep_prob = tf.placeholder(tf.float32, name="keep_prob", shape=[])
            return keep_prob

    @lazy_property
    def rec_keep_prob(self):
        with tf.variable_scope("rec_keep_prob"):
            rec_keep_prob = tf.placeholder(tf.float32, name="rec_keep_prob", shape=[])
            return rec_keep_prob

    @lazy_property
    def embeddings(self):
        with tf.variable_scope("embeddings"):
            embeddings = tf.get_variable(name="embeddings",
                                         initializer=tf.random_uniform(
                                             [self.input_size, self.embedding_size],
                                             -1.0,
                                             1.0),
                                         dtype=tf.float32)

            return embeddings

    @lazy_property
    def lookup(self):
        with tf.variable_scope("lookup"):
            lookup = tf.nn.embedding_lookup(self.embeddings, self.sentence)
            return lookup

    @lazy_property
    def biLSTM(self):
        for layer in range(self.num_layers):
            # avoid parameter sharing by defining new variable scopes for each layer
            with tf.variable_scope("biLSTM_layer_%d" % (layer + 1), reuse=tf.AUTO_REUSE):
                # forward cell
                layer_ltr_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                layer_ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
                    layer_ltr_cell,
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    state_keep_prob=self.rec_keep_prob
                )

                # backward cell
                layer_rtl_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                layer_rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
                    layer_rtl_cell,
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    state_keep_prob=self.rec_keep_prob
                )

                # actual bi-lstm for this layer takes the previous layer's outputs, concatenated
                (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    layer_ltr_cell,
                    layer_rtl_cell,
                    self.lookup if layer == 0 else lstm_outputs,    # input at the first layer is the lookup, otherwise previous outputs
                    sequence_length=self.seq_lengths,
                    dtype=tf.float32
                )

                # concat ltr_outputs and rtl_outputs to obtain the overall representation for this layer
                lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)

        return lstm_outputs

    @lazy_property
    def dense_fine(self):
        with tf.variable_scope("dense_fine"):
            weights_fine = tf.get_variable(
                name="weights_fine",
                shape=[2 * self.hidden_size, self.output_size],
                dtype=tf.float32
            )

            bias_fine = tf.get_variable(
                name="bias_fine",
                shape=[self.output_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_fine = tf.shape(self.biLSTM)[1]
            biLSTM_flat_fine = tf.reshape(self.biLSTM, [-1, 2 * self.hidden_size])
            logits_fine = biLSTM_flat_fine @ weights_fine + bias_fine
            logits_batch_fine = tf.reshape(logits_fine, [-1, max_lengths_fine, self.output_size])
            return logits_batch_fine

    @lazy_property
    def loss_fine(self):
        with tf.variable_scope("loss_fine"):
            losses_fine = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dense_fine, labels=self.labels)

            # exclude padding tokens from the gradient optimization
            mask_fine = tf.sequence_mask(self.seq_lengths)
            losses_fine = tf.boolean_mask(losses_fine, mask_fine)

            loss_fine = tf.reduce_mean(losses_fine)
            return loss_fine

    @lazy_property
    def loss(self):
        with tf.variable_scope("loss"):
            return self.loss_fine

    @lazy_property
    def train(self):
        with tf.variable_scope("train"):
            train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            return train

    def prepare_sentence(self, sentence, antivocab, output_vocab, labels=None):
        """
        Builds data structures for training and querying the BasicTagger model.
        If labels is specified, the model expects to be in training mode; otherwise: querying mode.

        :param sentence: List of XMLEntry objects produced by a TrainParser object
        :param antivocab: List of sub-sampled words
        :param output_vocab: Dictionary str -> int
        :param labels: List of GoldEntry objects produced by a GoldParser object (optional)
        :return: (sentence_input, labels_input, candidate_synsets)
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size

        def replacement_routine(l, entry):
            ret_word = None
            if l in antivocab:
                ret_word = output_vocab["<SUB>"]

            if entry.has_instance or ret_word is None:
                if l in output_vocab:
                    ret_word = output_vocab[l]
                elif ret_word is None:
                    ret_word = output_vocab["<UNK>"]

            return ret_word

        sentence_input = []
        labels_input = []
        candidate_synsets = []
        for xmlentry in sentence:
            iid = xmlentry.id
            lemma = xmlentry.lemma
            pos = xmlentry.pos

            sent_word = replacement_routine(l=lemma, entry=xmlentry)
            sentence_input.append(sent_word)
            if iid is None:
                labels_input.append(sent_word)

                # no instance word, just give the lemma itself as prediction
                candidates = [sent_word]
            else:
                iid = iid.split(".")[2][1:]
                iid = int(iid)

                if labels is not None:
                    sense = labels[iid].senses[0]
                    sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]
                    labels_input.append(sense)

                # fetch real synsets and get their mapping in the output vocabulary
                candidates = u.candidate_synsets(lemma, pos)
                candidates = [replacement_routine(c, ch.XMLEntry(id=None, lemma=c, pos="X", has_instance=True)) for c in candidates]

            candidate_synsets.append(candidates)

        return sentence_input, labels_input, candidate_synsets

    def batch_generator(self, batch_size, training_file_path, antivocab, output_vocab, gold_file_path=None, to_shuffle=True):
        """
        Builds batches for training and querying the BasicTagger model.
        If gold_file_path is provided, the model expects to be in training mode; otherwise: querying mode.

        :param batch_size: Size of batches to be created
        :param training_file_path: Path to the training corpus file (compliant to Raganato's format)
        :param antivocab: List of sub-sampled words
        :param output_vocab: Vocabulary to be used in the output layer
        :param gold_file_path: Path to the gold corpus file (compliant to Raganato's format) (optional)
        :param to_shuffle: True: shuffles batches (default); False: does not shuffle batches
        :return: (batch_sentences, batch_candidates, batch_labels, batch_lengths) if in training mode; (batch_sentences, batch_candidates, batch_lengths) if in querying mode
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size

        # for padding up to the longest sentence in the batch
        max_len = -1

        # holds sentences in the batch
        batch_sentences = []

        # holds labels in the batch if in training mode; None otherwise
        batch_labels = [] if gold_file_path is not None else None

        # holds candidate synsets for each sentence in the batch
        batch_candidates = []

        # holds lengths of sentences in the batch (excluding padding tokens)
        batch_lengths = []

        # candidate synsets for special tokens
        start_candidates = [output_vocab["<S>"]]
        end_candidates = [output_vocab["</S>"]]
        pad_candidates = [output_vocab["<PAD>"]]

        with \
                ch.TrainParser(file=training_file_path) as train_parser, \
                ch.GoldParser(file=(gold_file_path if gold_file_path is not None else "")) as gold_parser:

            for curr_sentence in train_parser.next_sentence():
                # check if the current sentence has an ambiguous word (prevents data.xml and gold.key.txt disalignment)
                has_instance = curr_sentence[0].has_instance

                # if a gold file is provided AND the sentence has at least one ambiguous word, get the labels
                curr_labels = None
                if gold_file_path is not None and has_instance:
                    curr_labels = next(gold_parser.next_sentence())

                if batch_labels is not None:
                    sent, lbls, candsyns = self.prepare_sentence(curr_sentence, antivocab, output_vocab, curr_labels)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    lbls = [output_vocab["<S>"]] + lbls + [output_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    batch_labels.append(lbls)
                    batch_candidates.append(candsyns)
                else:
                    sent, _, candsyns = self.prepare_sentence(curr_sentence, antivocab, output_vocab)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    batch_candidates.append(candsyns)

                # batch_size check
                if len(batch_sentences) == batch_size:
                    # align max_len to the next batch_size multiple
                    max_len = batch_size * ceil(float(max_len) / batch_size)

                    # apply padding
                    for i in range(len(batch_sentences)):
                        sent = batch_sentences[i]
                        candsyns = batch_candidates[i]

                        batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                        batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))

                        if batch_labels is not None:
                            lbls = batch_labels[i]
                            batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                    # shuffle if needed
                    if to_shuffle:
                        if batch_labels is not None:
                            batch_sentences, batch_labels, batch_candidates, batch_lengths = shuffle(batch_sentences,
                                                                                                     batch_labels,
                                                                                                     batch_candidates,
                                                                                                     batch_lengths)
                        else:
                            batch_sentences, batch_candidates, batch_lengths = shuffle(batch_sentences,
                                                                                       batch_candidates,
                                                                                       batch_lengths)

                    if batch_labels is not None:
                        yield batch_sentences, batch_candidates, batch_labels, batch_lengths
                    else:
                        yield batch_sentences, batch_candidates, batch_lengths

                    max_len = -1
                    batch_sentences = []
                    batch_labels = [] if gold_file_path is not None else None
                    batch_candidates = []
                    batch_lengths = []

            # handle incomplete batch in the same way
            if len(batch_sentences) > 0:
                # align max_len to the next batch_size multiple
                max_len = batch_size * ceil(float(max_len) / batch_size)

                # apply padding
                for i in range(len(batch_sentences)):
                    sent = batch_sentences[i]
                    candsyns = batch_candidates[i]

                    batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                    batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))
                    if batch_labels is not None:
                        lbls = batch_labels[i]
                        batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                # shuffle if needed
                if to_shuffle:
                    if batch_labels is not None:
                        batch_sentences, batch_labels, batch_candidates, batch_lengths = shuffle(batch_sentences,
                                                                                                 batch_labels,
                                                                                                 batch_candidates,
                                                                                                 batch_lengths)
                    else:
                        batch_sentences, batch_candidates, batch_lengths = shuffle(batch_sentences,
                                                                                   batch_candidates,
                                                                                   batch_lengths)

                if batch_labels is not None:
                    yield batch_sentences, batch_candidates, batch_labels, batch_lengths
                else:
                    yield batch_sentences, batch_candidates, batch_lengths

    def compute_accuracy(self, candidate_synsets, predictions, labels=None, return_predictions=False):
        """
        Computes accuracy depending on how many labels are correctly assigned, averaged.
        In case labels are not provided, the accuracy value has to be considered completely bogus.

        :param candidate_synsets: Words of the output vocabulary to be considered for each word when applying the argmax over the output probability distribution
        :param predictions: Labels predicted by the model
        :param labels: Gold labels (optional - MUST be provided in order to compute accuracy)
        :param return_predictions: True: also returns final predictions other than accuracy; False: only return accuracy (default)
        :return: accuracy value as float, in case return_predictions is False; (accuracy value as float, predictions as List of List), in case return_predictions is True
        """

        accuracy = 0.0
        tot_values = 0
        final_preds = []

        while len(predictions) > 0:
            #assert len(predictions) == len(labels), "Sizes of predictions and labels MUST match at every step"

            sent_pred = predictions[0]
            sent_candidates = candidate_synsets[0]
            if labels is not None:
                sent_labels = labels[0]
            real_preds = []

            # not needed anymore, free up some memory
            predictions = predictions[1:]
            candidate_synsets = candidate_synsets[1:]
            if labels is not None:
                labels = labels[1:]

            # iterate over each word in sentence to form predictions for the whole sentence
            for j in range(len(sent_pred)):
                single_word = sent_pred[j]
                single_candidates = sent_candidates[j]

                # only consider candidate synsets during argmax
                interesting_logits = single_word[single_candidates]
                single_pred = (np.argmax(interesting_logits)).astype(np.int32)
                real_preds.append(single_candidates[single_pred])

            # append to return final predictions, if needed
            if return_predictions:
                final_preds.append(real_preds)

            if labels is not None:
                # strip both predictions and labels of special tokens and padding
                non_padding = np.count_nonzero(sent_labels)
                real_preds = real_preds[1:non_padding - 1]
                sent_labels = sent_labels[1:non_padding - 1]

                real_preds = np.asarray(real_preds)
                sent_labels = np.asarray(sent_labels)

                accuracy += np.count_nonzero(real_preds == sent_labels) * 1.0
                tot_values += non_padding - 2
            else:
                # avoid division by zero error
                accuracy = 0
                tot_values = 1

        if return_predictions:
            return accuracy / tot_values, final_preds
        else:
            return accuracy / tot_values


class MultitaskTagger(BasicTagger):
    """
    BasicTagger extended to perform WSD for fine-grained senses exploiting coarse-grained senses and POS tagging jointly (if specified).
    In case both pos_size and coarse_size are not defined, this model has the same behaviour of BasicTagger.
    """

    def __init__(self, learning_rate, embedding_size, hidden_size, num_layers, input_size, output_size, pos_size=None, coarse_size=None):
        """
        Initializes a multi-layer Bi-LSTM model for WSD as means of sequence tagging.

        :param learning_rate: Learning rate to be used in the optimization
        :param embedding_size: Embedding size for the input vocabulary
        :param hidden_size: Hidden size for the LSTM networks
        :param num_layers: Number of LSTM layers to stack
        :param input_size: Size of the input vocabulary (plain words)
        :param output_size: Size of the output vocabulary (fine-grained senses)
        :param pos_size: Size of the POS vocabulary (optional)
        :param coarse_size: Size of coarse-grained sense vocabulary (optional)
        """

        assert num_layers > 0, "Layers number must be at least 1"

        self.lr = learning_rate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.pos_size = pos_size
        self.coarse_size = coarse_size

        # correctly initialize graph
        self.sentence
        self.labels
        if self.pos_size is not None:
            self.labels_pos
        if self.coarse_size is not None:
            self.labels_coarse
        self.seq_lengths

        self.keep_prob
        self.rec_keep_prob

        self.embeddings
        self.lookup

        self.biLSTM

        self.dense_fine
        if self.pos_size is not None:
            self.dense_pos
        if self.coarse_size is not None:
            self.dense_coarse

        self.loss_fine
        if self.pos_size is not None:
            self.loss_pos
        if self.coarse_size is not None:
            self.loss_coarse

        self.loss
        self.train

    @lazy_property
    def labels_pos(self):
        with tf.variable_scope("labels_pos"):
            labels_pos = tf.placeholder(tf.int32, name="labels_pos", shape=[None, None])
            return labels_pos

    @lazy_property
    def labels_coarse(self):
        with tf.variable_scope("labels_coarse"):
            labels_coarse = tf.placeholder(tf.int32, name="labels_coarse", shape=[None, None])
            return labels_coarse

    @lazy_property
    def dense_pos(self):
        with tf.variable_scope("dense_pos"):
            weights_pos = tf.get_variable(
                name="weights_pos",
                shape=[2 * self.hidden_size, self.pos_size],
                dtype=tf.float32
            )

            bias_pos = tf.get_variable(
                name="bias_pos",
                shape=[self.pos_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_pos = tf.shape(self.biLSTM)[1]
            biLSTM_flat_pos = tf.reshape(self.biLSTM, [-1, 2 * self.hidden_size])
            logits_pos = biLSTM_flat_pos @ weights_pos + bias_pos
            logits_batch_pos = tf.reshape(logits_pos, [-1, max_lengths_pos, self.pos_size])
            return logits_batch_pos

    @lazy_property
    def dense_coarse(self):
        with tf.variable_scope("dense_coarse"):
            weights_coarse = tf.get_variable(
                name="weights_coarse",
                shape=[2 * self.hidden_size, self.coarse_size],
                dtype=tf.float32
            )

            bias_coarse = tf.get_variable(
                name="bias_coarse",
                shape=[self.coarse_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_coarse = tf.shape(self.biLSTM)[1]
            biLSTM_flat_coarse = tf.reshape(self.biLSTM, [-1, 2 * self.hidden_size])
            logits_coarse = biLSTM_flat_coarse @ weights_coarse + bias_coarse
            logits_batch_coarse = tf.reshape(logits_coarse, [-1, max_lengths_coarse, self.coarse_size])
            return logits_batch_coarse

    @lazy_property
    def loss_pos(self):
        with tf.variable_scope("loss_pos"):
            losses_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dense_pos, labels=self.labels_pos)

            # exclude padding tokens from the gradient optimization
            mask_pos = tf.sequence_mask(self.seq_lengths)
            losses_pos = tf.boolean_mask(losses_pos, mask_pos)

            loss_pos = tf.reduce_mean(losses_pos)
            return loss_pos

    @lazy_property
    def loss_coarse(self):
        with tf.variable_scope("loss_coarse"):
            losses_coarse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dense_coarse, labels=self.labels_coarse)

            # exclude padding tokens from the gradient optimization
            mask_coarse = tf.sequence_mask(self.seq_lengths)
            losses_coarse = tf.boolean_mask(losses_coarse, mask_coarse)

            loss_coarse = tf.reduce_mean(losses_coarse)
            return loss_coarse

    @lazy_property
    def loss(self):
        with tf.variable_scope("loss"):
            loss = self.loss_fine

            if self.pos_size is not None:
                loss += self.loss_pos
            if self.coarse_size is not None:
                loss += self.loss_coarse

            return loss

    def prepare_sentence(self, sentence, antivocab, output_vocab, pos_vocab=None, coarse_vocab=None, wn2bn=None, bn2coarse=None, labels=None):
        """
        Builds data structures for training and querying the MultitaskTagger model.
        If labels is specified, the model expects to be in training mode; otherwise: querying mode.

        :param sentence: List of XMLEntry objects produced by a TrainParser object
        :param antivocab: List of sub-sampled words
        :param output_vocab: Dictionary str -> int for fine-grained senses
        :param pos_vocab: Dictionary str -> int for POS tags
        :param coarse_vocab: Dictionary str -> int for coarse-grained senses
        :param wn2bn: Dictionary str -> List of str for mapping WordNet IDs to BabelNet IDs
        :param bn2coarse: Dictionary str -> List of str for mapping BabelNet IDs to the coarse-grained sense vocabulary of choice
        :param labels: List of GoldEntry objects produced by a GoldParser object (optional)
        :return: (sentence_input, pos_input, coarse_input, labels_input, candidate_synsets)
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size
        assert self.pos_size is None or (self.pos_size is not None and pos_vocab is not None and len(pos_vocab) == self.pos_size)
        assert self.coarse_size is None or (self.coarse_size is not None and coarse_vocab is not None and len(coarse_vocab) == self.coarse_size)
        assert self.coarse_size is None or (self.coarse_size is not None and coarse_vocab is not None and wn2bn is not None and bn2coarse is not None)

        def replacement_routine(l, entry):
            ret_word = None
            if l in antivocab:
                ret_word = output_vocab["<SUB>"]

            if entry.has_instance or ret_word is None:
                if l in output_vocab:
                    ret_word = output_vocab[l]
                elif ret_word is None:
                    ret_word = output_vocab["<UNK>"]

            return ret_word

        sentence_input = []
        pos_input = []
        coarse_input = []
        labels_input = []
        candidate_synsets = []
        for xmlentry in sentence:
            iid = xmlentry.id
            lemma = xmlentry.lemma
            pos = xmlentry.pos

            sent_word = replacement_routine(l=lemma, entry=xmlentry)
            sentence_input.append(sent_word)
            if self.pos_size is not None:
                pos_input.append(pos_vocab[pos] if pos in pos_vocab else pos_vocab["<UNK>"])

            if iid is None:
                labels_input.append(sent_word)

                # in case the word is not an instance word, exploit SUB tag for its coarse sense
                if self.coarse_size is not None:
                    coarse_input.append(coarse_vocab["<SUB>"])

                # no instance word, just give the lemma itself as prediction
                candidates = [sent_word]
            else:
                iid = iid.split(".")[2][1:]
                iid = int(iid)

                if labels is not None:
                    # WordNet ID
                    sense = labels[iid].senses[0]

                    # get the coarse sense associated
                    if self.coarse_size is not None:
                        bn_id = wn2bn[sense][0]
                        coarse_sense = bn2coarse.get(bn_id, ["factotum"])[0]     # fixes missing BabelNet IDs in the mapping to WN Domains (does not happen for LEX)
                        coarse_input.append(coarse_vocab[coarse_sense] if coarse_sense in coarse_vocab else coarse_vocab["<UNK>"])

                    sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]
                    labels_input.append(sense)

                # fetch real synsets and get their mapping in the output vocabulary
                candidates = u.candidate_synsets(lemma, pos)
                candidates = [replacement_routine(c, ch.XMLEntry(id=None, lemma=c, pos="X", has_instance=True)) for c in candidates]

            candidate_synsets.append(candidates)

        return sentence_input, pos_input, coarse_input, labels_input, candidate_synsets

    def batch_generator(self, batch_size, training_file_path, antivocab, output_vocab, pos_vocab=None, coarse_vocab=None, wn2bn=None, bn2coarse=None, gold_file_path=None, to_shuffle=True):
        """
        Builds batches for training and querying the MultitaskTagger model.
        If gold_file_path is provided, the model expects to be in training mode; otherwise: querying mode.

        :param batch_size: Size of batches to be created
        :param training_file_path: Path to the training corpus file (compliant to Raganato's format)
        :param antivocab: List of sub-sampled words
        :param output_vocab: Vocabulary to be used in the output layer
        :param pos_vocab: Dictionary str -> int for POS tags
        :param coarse_vocab: Dictionary str -> int for coarse-grained senses
        :param wn2bn: Dictionary str -> List of str for mapping WordNet IDs to BabelNet IDs
        :param bn2coarse: Dictionary str -> List of str for mapping BabelNet IDs to the coarse-grained sense vocabulary of choice
        :param gold_file_path: Path to the gold corpus file (compliant to Raganato's format) (optional)
        :param to_shuffle: True: shuffles batches (default); False: does not shuffle batches
        :return: (batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_labels, batch_lengths) if in training mode;
                (batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_lengths) if in querying mode
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size
        assert self.pos_size is None or (self.pos_size is not None and pos_vocab is not None and len(pos_vocab) == self.pos_size)
        assert self.coarse_size is None or (self.coarse_size is not None and coarse_vocab is not None and len(coarse_vocab) == self.coarse_size)
        assert self.coarse_size is None or (self.coarse_size is not None and coarse_vocab is not None and wn2bn is not None and bn2coarse is not None)

        # for padding up to the longest sentence in the batch
        max_len = -1

        # holds sentences in the batch
        batch_sentences = []

        # holds POS tags in the batch if needed; None otherwise
        batch_pos = [] if self.pos_size is not None else None

        # holds coarse-grained senses in the batch if needed; None otherwise
        batch_coarse = [] if self.coarse_size is not None else None

        # holds labels in the batch if in training mode; None otherwise
        batch_labels = [] if gold_file_path is not None else None

        # holds candidate synsets for each sentence in the batch
        batch_candidates = []

        # holds lengths of sentences in the batch (excluding padding tokens)
        batch_lengths = []

        # candidate synsets for special tokens
        start_candidates = [output_vocab["<S>"]]
        end_candidates = [output_vocab["</S>"]]
        pad_candidates = [output_vocab["<PAD>"]]

        with \
                ch.TrainParser(file=training_file_path) as train_parser, \
                ch.GoldParser(file=(gold_file_path if gold_file_path is not None else "")) as gold_parser:

            for curr_sentence in train_parser.next_sentence():
                # check if the current sentence has an ambiguous word (prevents data.xml and gold.key.txt disalignment)
                has_instance = curr_sentence[0].has_instance

                # if a gold file is provided AND the sentence has at least one ambiguous word, get the labels
                curr_labels = None
                if gold_file_path is not None and has_instance:
                    curr_labels = next(gold_parser.next_sentence())

                if batch_labels is not None:
                    sent, sent_pos, sent_coarse, lbls, candsyns = self.prepare_sentence(curr_sentence,
                                                                                        antivocab,
                                                                                        output_vocab,
                                                                                        pos_vocab,
                                                                                        coarse_vocab,
                                                                                        wn2bn,
                                                                                        bn2coarse,
                                                                                        curr_labels)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    if batch_pos is not None:
                        sent_pos = [pos_vocab["<S>"]] + sent_pos + [pos_vocab["</S>"]]
                    if batch_coarse is not None:
                        sent_coarse = [coarse_vocab["<S>"]] + sent_coarse + [coarse_vocab["</S>"]]
                    lbls = [output_vocab["<S>"]] + lbls + [output_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    if batch_pos is not None:
                        batch_pos.append(sent_pos)
                    if batch_coarse is not None:
                        batch_coarse.append(sent_coarse)
                    batch_labels.append(lbls)
                    batch_candidates.append(candsyns)
                else:
                    sent, sent_pos, sent_coarse, _, candsyns = self.prepare_sentence(curr_sentence,
                                                                                     antivocab,
                                                                                     output_vocab,
                                                                                     pos_vocab,
                                                                                     coarse_vocab,
                                                                                     wn2bn,
                                                                                     bn2coarse)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    if batch_pos is not None:
                        sent_pos = [pos_vocab["<S>"]] + sent_pos + [pos_vocab["</S>"]]
                    if batch_coarse is not None:
                        sent_coarse = [coarse_vocab["<S>"]] + sent_coarse + [coarse_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    if batch_pos is not None:
                        batch_pos.append(sent_pos)
                    if batch_coarse is not None:
                        batch_coarse.append(sent_coarse)
                    batch_candidates.append(candsyns)

                # batch_size check
                if len(batch_sentences) == batch_size:
                    # align max_len to the next batch_size multiple
                    max_len = batch_size * ceil(float(max_len) / batch_size)

                    # apply padding
                    for i in range(len(batch_sentences)):
                        sent = batch_sentences[i]
                        candsyns = batch_candidates[i]

                        batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                        if batch_pos is not None:
                            sent_pos = batch_pos[i]
                            batch_pos[i] = sent_pos + [pos_vocab["<PAD>"]] * (max_len - len(sent))
                        if batch_coarse is not None:
                            sent_coarse = batch_coarse[i]
                            batch_coarse[i] = sent_coarse + [coarse_vocab["<PAD>"]] * (max_len - len(sent))

                        batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))

                        if batch_labels is not None:
                            lbls = batch_labels[i]
                            batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                    # shuffle if needed
                    if to_shuffle:
                        if batch_labels is not None:
                            if batch_pos is None and batch_coarse is None:
                                batch_sentences, \
                                    batch_labels, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_labels,
                                                            batch_candidates,
                                                            batch_lengths)
                            elif batch_pos is not None and batch_coarse is None:
                                batch_sentences, \
                                    batch_pos, \
                                    batch_labels, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_pos,
                                                            batch_labels,
                                                            batch_candidates,
                                                            batch_lengths)
                            elif batch_pos is None and batch_coarse is not None:
                                batch_sentences, \
                                    batch_coarse, \
                                    batch_labels, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_coarse,
                                                            batch_labels,
                                                            batch_candidates,
                                                            batch_lengths)
                            else:
                                batch_sentences, \
                                    batch_pos, \
                                    batch_coarse, \
                                    batch_labels, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_pos,
                                                            batch_coarse,
                                                            batch_labels,
                                                            batch_candidates,
                                                            batch_lengths)
                        else:
                            if batch_pos is None and batch_coarse is None:
                                batch_sentences, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_candidates,
                                                            batch_lengths)
                            elif batch_pos is not None and batch_coarse is None:
                                batch_sentences, \
                                    batch_pos, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_pos,
                                                            batch_candidates,
                                                            batch_lengths)
                            elif batch_pos is None and batch_coarse is not None:
                                batch_sentences, \
                                    batch_coarse, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_coarse,
                                                            batch_candidates,
                                                            batch_lengths)
                            else:
                                batch_sentences, \
                                    batch_pos, \
                                    batch_coarse, \
                                    batch_candidates, \
                                    batch_lengths = shuffle(batch_sentences,
                                                            batch_pos,
                                                            batch_coarse,
                                                            batch_candidates,
                                                            batch_lengths)

                    if batch_labels is not None:
                        yield batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_labels, batch_lengths
                    else:
                        yield batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_lengths

                    max_len = -1
                    batch_sentences = []
                    batch_pos = [] if self.pos_size is not None else None
                    batch_coarse = [] if self.coarse_size is not None else None
                    batch_labels = [] if gold_file_path is not None else None
                    batch_candidates = []
                    batch_lengths = []

            # handle incomplete batch in the same way
            if len(batch_sentences) > 0:
                # align max_len to the next batch_size multiple
                max_len = batch_size * ceil(float(max_len) / batch_size)

                # apply padding
                for i in range(len(batch_sentences)):
                    sent = batch_sentences[i]
                    candsyns = batch_candidates[i]

                    batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                    if batch_pos is not None:
                        sent_pos = batch_pos[i]
                        batch_pos[i] = sent_pos + [pos_vocab["<PAD>"]] * (max_len - len(sent))
                    if batch_coarse is not None:
                        sent_coarse = batch_coarse[i]
                        batch_coarse[i] = sent_coarse + [coarse_vocab["<PAD>"]] * (max_len - len(sent))

                    batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))

                    if batch_labels is not None:
                        lbls = batch_labels[i]
                        batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                # shuffle if needed
                if to_shuffle:
                    if batch_labels is not None:
                        if batch_pos is None and batch_coarse is None:
                            batch_sentences, \
                                batch_labels, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_labels,
                                                        batch_candidates,
                                                        batch_lengths)
                        elif batch_pos is not None and batch_coarse is None:
                            batch_sentences, \
                                batch_pos, \
                                batch_labels, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_labels,
                                                        batch_candidates,
                                                        batch_lengths)
                        elif batch_pos is None and batch_coarse is not None:
                            batch_sentences, \
                                batch_coarse, \
                                batch_labels, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_coarse,
                                                        batch_labels,
                                                        batch_candidates,
                                                        batch_lengths)
                        else:
                            batch_sentences, \
                                batch_pos, \
                                batch_coarse, \
                                batch_labels, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_coarse,
                                                        batch_labels,
                                                        batch_candidates,
                                                        batch_lengths)
                    else:
                        if batch_pos is None and batch_coarse is None:
                            batch_sentences, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_candidates,
                                                        batch_lengths)
                        elif batch_pos is not None and batch_coarse is None:
                            batch_sentences, \
                                batch_pos, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_candidates,
                                                        batch_lengths)
                        elif batch_pos is None and batch_coarse is not None:
                            batch_sentences, \
                                batch_coarse, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_coarse,
                                                        batch_candidates,
                                                        batch_lengths)
                        else:
                            batch_sentences, \
                                batch_pos, \
                                batch_coarse, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_coarse,
                                                        batch_candidates,
                                                        batch_lengths)

                if batch_labels is not None:
                    yield batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_labels, batch_lengths
                else:
                    yield batch_sentences, batch_pos, batch_coarse, batch_candidates, batch_lengths


class DomainAwareTagger(MultitaskTagger):
    """
    MultitaskTagger extended to perform WSD considering the whole sentence's domain.
    This should force the model to internalize a domain's representation and exploit it during future WSD tagging tasks.

    While this model could work supplying Lexnames as coarse-grained senses, it is recommended to use WordNet Domains,
    as the latter ones are more suitable to identify a whole sentence's domain.
    """

    def __init__(self, learning_rate, embedding_size, hidden_size, num_layers, input_size, output_size, pos_size, coarse_size, swap_states=False):
        """
        Initializes a multi-layer Bi-LSTM model for WSD as means of sequence tagging.

        :param learning_rate: Learning rate to be used in the optimization
        :param embedding_size: Embedding size for the input vocabulary
        :param hidden_size: Hidden size for the LSTM networks
        :param num_layers: Number of LSTM layers to stack
        :param input_size: Size of the input vocabulary (plain words)
        :param output_size: Size of the output vocabulary (fine-grained senses)
        :param pos_size: Size of the POS vocabulary
        :param coarse_size: Size of coarse-grained sense vocabulary
        :param swap_states: True: swaps forward and backward hidden states from a BiLSTM layer to the next one;
                            False: previous forward and backward hidden states are fed as forward and backward initial
                                    states, respectively (default)
        """

        assert num_layers > 0, "Layers number must be at least 1"

        self.lr = learning_rate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.pos_size = pos_size
        self.coarse_size = coarse_size
        self.swap_states = swap_states

        # correctly initialize graph
        self.sentence
        self.labels
        self.labels_pos
        self.labels_coarse
        self.label_domain
        self.seq_lengths

        self.keep_prob
        self.rec_keep_prob

        self.embeddings
        self.domain_embeddings
        self.lookup
        self.domain_lookup

        self.biLSTM

        self.dense_fine
        self.dense_pos
        self.dense_coarse

        self.loss_fine
        self.loss_pos
        self.loss_coarse
        self.loss_domain

        self.loss
        self.train

    @lazy_property
    def label_domain(self):
        with tf.variable_scope("label_domain"):
            label_domain = tf.placeholder(tf.int32, name="label_domain", shape=[None])
            return label_domain

    @lazy_property
    def domain_embeddings(self):
        with tf.variable_scope("domain_embeddings"):
            domain_embeddings = tf.get_variable(name="domain_embeddings",
                                                initializer=tf.random_uniform(
                                                    [self.coarse_size, 2 * self.hidden_size],
                                                    -1.0,
                                                    1.0),
                                                dtype=tf.float32)

            return domain_embeddings

    @lazy_property
    def domain_lookup(self):
        with tf.variable_scope("domain_lookup"):
            domain_lookup = tf.nn.embedding_lookup(self.domain_embeddings, self.label_domain)
            return domain_lookup

    @lazy_property
    def biLSTM(self):
        for layer in range(self.num_layers):
            # avoid parameter sharing by defining new variable scopes for each layer
            with tf.variable_scope("biLSTM_layer_%d" % (layer + 1), reuse=tf.AUTO_REUSE):
                # forward cell
                layer_ltr_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                layer_ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
                    layer_ltr_cell,
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    state_keep_prob=self.rec_keep_prob
                )

                # backward cell
                layer_rtl_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                layer_rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
                    layer_rtl_cell,
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    state_keep_prob=self.rec_keep_prob
                )

                # actual bi-lstm for this layer takes the previous layer's outputs, concatenated
                (ltr_outputs, rtl_outputs), (ltr_hidden, rtl_hidden) = tf.nn.bidirectional_dynamic_rnn(
                    layer_ltr_cell,
                    layer_rtl_cell,
                    # input at the first layer is the lookup, otherwise previous outputs
                    self.lookup if layer == 0 else lstm_outputs,
                    sequence_length=self.seq_lengths,
                    # initial states set to the previous layer's hidden states, swapped if needed
                    initial_state_fw=None if layer == 0 else (rtl_hidden if self.swap_states else ltr_hidden),
                    initial_state_bw=None if layer == 0 else (ltr_hidden if self.swap_states else rtl_hidden),
                    dtype=tf.float32
                )

                # concat ltr_outputs and rtl_outputs to obtain the overall representation for this layer
                lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)

                # concat last states of ltr_hidden and rtl_hidden to obtain the internal representation of the sentence
                lstm_hidden = tf.concat([ltr_hidden.c, rtl_hidden.c], axis=-1)

        return {"biLSTM_OUT": lstm_outputs, "biLSTM_HIDDEN": lstm_hidden}

    @lazy_property
    def dense_fine(self):
        with tf.variable_scope("dense_fine"):
            weights_fine = tf.get_variable(
                name="weights_fine",
                shape=[2 * self.hidden_size, self.output_size],
                dtype=tf.float32
            )

            bias_fine = tf.get_variable(
                name="bias_fine",
                shape=[self.output_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_fine = tf.shape(self.biLSTM["biLSTM_OUT"])[1]
            biLSTM_flat_fine = tf.reshape(self.biLSTM["biLSTM_OUT"], [-1, 2 * self.hidden_size])
            logits_fine = biLSTM_flat_fine @ weights_fine + bias_fine
            logits_batch_fine = tf.reshape(logits_fine, [-1, max_lengths_fine, self.output_size])
            return logits_batch_fine

    @lazy_property
    def dense_pos(self):
        with tf.variable_scope("dense_pos"):
            weights_pos = tf.get_variable(
                name="weights_pos",
                shape=[2 * self.hidden_size, self.pos_size],
                dtype=tf.float32
            )

            bias_pos = tf.get_variable(
                name="bias_pos",
                shape=[self.pos_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_pos = tf.shape(self.biLSTM["biLSTM_OUT"])[1]
            biLSTM_flat_pos = tf.reshape(self.biLSTM["biLSTM_OUT"], [-1, 2 * self.hidden_size])
            logits_pos = biLSTM_flat_pos @ weights_pos + bias_pos
            logits_batch_pos = tf.reshape(logits_pos, [-1, max_lengths_pos, self.pos_size])
            return logits_batch_pos

    @lazy_property
    def dense_coarse(self):
        with tf.variable_scope("dense_coarse"):
            weights_coarse = tf.get_variable(
                name="weights_coarse",
                shape=[2 * self.hidden_size, self.coarse_size],
                dtype=tf.float32
            )

            bias_coarse = tf.get_variable(
                name="bias_coarse",
                shape=[self.coarse_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            max_lengths_coarse = tf.shape(self.biLSTM["biLSTM_OUT"])[1]
            biLSTM_flat_coarse = tf.reshape(self.biLSTM["biLSTM_OUT"], [-1, 2 * self.hidden_size])
            logits_coarse = biLSTM_flat_coarse @ weights_coarse + bias_coarse
            logits_batch_coarse = tf.reshape(logits_coarse, [-1, max_lengths_coarse, self.coarse_size])
            return logits_batch_coarse

    @lazy_property
    def loss_domain(self):
        with tf.variable_scope("loss_domain"):
            # normalize vectors
            domain_lookup = tf.nn.l2_normalize(self.domain_lookup, axis=-1)
            hidden_state = tf.nn.l2_normalize(self.biLSTM["biLSTM_HIDDEN"], axis=-1)

            # cosine distance between the real domain embeddings and the last hidden state of the biLSTM
            loss_domain = tf.losses.cosine_distance(domain_lookup,
                                                    hidden_state,
                                                    axis=-1)

            return loss_domain

    @lazy_property
    def loss(self):
        with tf.variable_scope("loss"):
            loss = self.loss_fine + self.loss_pos + self.loss_coarse + self.loss_domain
            return loss

    def prepare_sentence(self, sentence, antivocab, output_vocab, pos_vocab, coarse_vocab, wn2bn, bn2coarse, labels=None):
        """
        Builds data structures for training and querying the DomainAwareTagger model.
        If labels is specified, the model expects to be in training mode; otherwise: querying mode.

        :param sentence: List of XMLEntry objects produced by a TrainParser object
        :param antivocab: List of sub-sampled words
        :param output_vocab: Dictionary str -> int for fine-grained senses
        :param pos_vocab: Dictionary str -> int for POS tags
        :param coarse_vocab: Dictionary str -> int for coarse-grained senses
        :param wn2bn: Dictionary str -> List of str for mapping WordNet IDs to BabelNet IDs
        :param bn2coarse: Dictionary str -> List of str for mapping BabelNet IDs to the coarse-grained sense vocabulary of choice
        :param labels: List of GoldEntry objects produced by a GoldParser object (optional)
        :return: (sentence_input, pos_input, coarse_input, domain_input, labels_input, candidate_synsets)
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size
        assert len(pos_vocab) == self.pos_size
        assert len(coarse_vocab) == self.coarse_size

        def replacement_routine(l, entry):
            ret_word = None
            if l in antivocab:
                ret_word = output_vocab["<SUB>"]

            if entry.has_instance or ret_word is None:
                if l in output_vocab:
                    ret_word = output_vocab[l]
                elif ret_word is None:
                    ret_word = output_vocab["<UNK>"]

            return ret_word

        sentence_input = []
        pos_input = []
        coarse_input = []
        labels_input = []
        domain_occurrences = {}
        candidate_synsets = []
        for xmlentry in sentence:
            iid = xmlentry.id
            lemma = xmlentry.lemma
            pos = xmlentry.pos

            sent_word = replacement_routine(l=lemma, entry=xmlentry)
            sentence_input.append(sent_word)
            pos_input.append(pos_vocab[pos] if pos in pos_vocab else pos_vocab["<UNK>"])

            if iid is None:
                labels_input.append(sent_word)

                # in case the word is not an instance word, exploit SUB tag for its coarse sense
                coarse_sense = coarse_vocab["<SUB>"]
                coarse_input.append(coarse_sense)

                # store in the dictionary to compute the majority domain
                domain_occurrences[coarse_sense] = domain_occurrences.get(coarse_sense, 0) + 1

                # no instance word, just give the lemma itself as prediction
                candidates = [sent_word]
            else:
                iid = iid.split(".")[2][1:]
                iid = int(iid)

                if labels is not None:
                    # WordNet ID
                    sense = labels[iid].senses[0]

                    # get the coarse sense associated
                    bn_id = wn2bn[sense][0]
                    coarse_sense = bn2coarse.get(bn_id, ["factotum"])[0]     # fixes missing BabelNet IDs in the mapping to WN Domains (does not happen for LEX)
                    coarse_sense = coarse_vocab[coarse_sense] if coarse_sense in coarse_vocab else coarse_vocab["<UNK>"]
                    coarse_input.append(coarse_sense)

                    # store in the dictionary to compute the majority domain
                    domain_occurrences[coarse_sense] = domain_occurrences.get(coarse_sense, 0) + 1

                    sense = output_vocab[sense] if sense in output_vocab else output_vocab["<UNK>"]
                    labels_input.append(sense)

                # fetch real synsets and get their mapping in the output vocabulary
                candidates = u.candidate_synsets(lemma, pos)
                candidates = [replacement_routine(c, ch.XMLEntry(id=None, lemma=c, pos="X", has_instance=True)) for c in candidates]

            candidate_synsets.append(candidates)

        # search for the majority domain
        domain_input = None
        if len(domain_occurrences) > 0:
            domain_occurrences = sorted(domain_occurrences.items(), key=lambda x: x[1], reverse=True)
            domain_input = domain_occurrences[0][0]
        if domain_input is None:
            domain_input = coarse_vocab["<UNK>"]

        return sentence_input, pos_input, coarse_input, domain_input, labels_input, candidate_synsets

    def batch_generator(self, batch_size, training_file_path, antivocab, output_vocab, pos_vocab, coarse_vocab, wn2bn, bn2coarse, gold_file_path=None, to_shuffle=True):
        """
        Builds batches for training and querying the DomainAwareTagger model.
        If gold_file_path is provided, the model expects to be in training mode; otherwise: querying mode.

        :param batch_size: Size of batches to be created
        :param training_file_path: Path to the training corpus file (compliant to Raganato's format)
        :param antivocab: List of sub-sampled words
        :param output_vocab: Vocabulary to be used in the output layer
        :param pos_vocab: Dictionary str -> int for POS tags
        :param coarse_vocab: Dictionary str -> int for coarse-grained senses
        :param wn2bn: Dictionary str -> List of str for mapping WordNet IDs to BabelNet IDs
        :param bn2coarse: Dictionary str -> List of str for mapping BabelNet IDs to the coarse-grained sense vocabulary of choice
        :param gold_file_path: Path to the gold corpus file (compliant to Raganato's format) (optional)
        :param to_shuffle: True: shuffles batches (default); False: does not shuffle batches
        :return: (batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_labels, batch_lengths) if in training mode;
                (batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_lengths) if in querying mode
        """

        # check vocabulary alignment with the output layer
        assert len(output_vocab) == self.output_size
        assert len(pos_vocab) == self.pos_size
        assert len(coarse_vocab) == self.coarse_size

        # for padding up to the longest sentence in the batch
        max_len = -1

        # holds sentences in the batch
        batch_sentences = []

        # holds POS tags in the batch
        batch_pos = []

        # holds coarse-grained senses in the batch
        batch_coarse = []

        # holds labels in the batch if in training mode; None otherwise
        batch_labels = [] if gold_file_path is not None else None

        # holds sentence domains in the batch
        batch_domains = []

        # holds candidate synsets for each sentence in the batch
        batch_candidates = []

        # holds lengths of sentences in the batch (excluding padding tokens)
        batch_lengths = []

        # candidate synsets for special tokens
        start_candidates = [output_vocab["<S>"]]
        end_candidates = [output_vocab["</S>"]]
        pad_candidates = [output_vocab["<PAD>"]]

        with \
                ch.TrainParser(file=training_file_path) as train_parser, \
                ch.GoldParser(file=(gold_file_path if gold_file_path is not None else "")) as gold_parser:

            for curr_sentence in train_parser.next_sentence():
                # check if the current sentence has an ambiguous word (prevents data.xml and gold.key.txt disalignment)
                has_instance = curr_sentence[0].has_instance

                # if a gold file is provided AND the sentence has at least one ambiguous word, get the labels
                curr_labels = None
                if gold_file_path is not None and has_instance:
                    curr_labels = next(gold_parser.next_sentence())

                if batch_labels is not None:
                    sent, sent_pos, sent_coarse, sent_domain, lbls, candsyns = self.prepare_sentence(curr_sentence,
                                                                                                     antivocab,
                                                                                                     output_vocab,
                                                                                                     pos_vocab,
                                                                                                     coarse_vocab,
                                                                                                     wn2bn,
                                                                                                     bn2coarse,
                                                                                                     curr_labels)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    sent_pos = [pos_vocab["<S>"]] + sent_pos + [pos_vocab["</S>"]]
                    sent_coarse = [coarse_vocab["<S>"]] + sent_coarse + [coarse_vocab["</S>"]]
                    lbls = [output_vocab["<S>"]] + lbls + [output_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    batch_pos.append(sent_pos)
                    batch_coarse.append(sent_coarse)
                    batch_labels.append(lbls)
                    batch_domains.append(sent_domain)
                    batch_candidates.append(candsyns)
                else:
                    sent, sent_pos, sent_coarse, sent_domain, _, candsyns = self.prepare_sentence(curr_sentence,
                                                                                                  antivocab,
                                                                                                  output_vocab,
                                                                                                  pos_vocab,
                                                                                                  coarse_vocab,
                                                                                                  wn2bn,
                                                                                                  bn2coarse)

                    # add sentence start and end tokens
                    sent = [output_vocab["<S>"]] + sent + [output_vocab["</S>"]]
                    sent_pos = [pos_vocab["<S>"]] + sent_pos + [pos_vocab["</S>"]]
                    sent_coarse = [coarse_vocab["<S>"]] + sent_coarse + [coarse_vocab["</S>"]]
                    candsyns = [start_candidates] + candsyns + [end_candidates]

                    curr_len = len(sent)
                    batch_lengths.append(curr_len)
                    if max_len == -1 or curr_len > max_len:
                        max_len = curr_len

                    batch_sentences.append(sent)
                    batch_pos.append(sent_pos)
                    batch_coarse.append(sent_coarse)
                    batch_domains.append(sent_domain)
                    batch_candidates.append(candsyns)

                # batch_size check
                if len(batch_sentences) == batch_size:
                    # align max_len to the next batch_size multiple
                    max_len = batch_size * ceil(float(max_len) / batch_size)

                    # apply padding
                    for i in range(len(batch_sentences)):
                        sent = batch_sentences[i]
                        candsyns = batch_candidates[i]

                        batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                        sent_pos = batch_pos[i]
                        batch_pos[i] = sent_pos + [pos_vocab["<PAD>"]] * (max_len - len(sent))
                        sent_coarse = batch_coarse[i]
                        batch_coarse[i] = sent_coarse + [coarse_vocab["<PAD>"]] * (max_len - len(sent))
                        batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))

                        if batch_labels is not None:
                            lbls = batch_labels[i]
                            batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                    # shuffle if needed
                    if to_shuffle:
                        if batch_labels is not None:
                            batch_sentences, \
                                batch_pos, \
                                batch_coarse, \
                                batch_labels, \
                                batch_domains, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_coarse,
                                                        batch_labels,
                                                        batch_domains,
                                                        batch_candidates,
                                                        batch_lengths)
                        else:
                            batch_sentences, \
                                batch_pos, \
                                batch_coarse, \
                                batch_domains, \
                                batch_candidates, \
                                batch_lengths = shuffle(batch_sentences,
                                                        batch_pos,
                                                        batch_coarse,
                                                        batch_domains,
                                                        batch_candidates,
                                                        batch_lengths)

                    if batch_labels is not None:
                        yield batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_labels, batch_lengths
                    else:
                        yield batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_lengths

                    max_len = -1
                    batch_sentences = []
                    batch_pos = []
                    batch_coarse = []
                    batch_labels = [] if gold_file_path is not None else None
                    batch_domains = []
                    batch_candidates = []
                    batch_lengths = []

            # handle incomplete batch in the same way
            if len(batch_sentences) > 0:
                # align max_len to the next batch_size multiple
                max_len = batch_size * ceil(float(max_len) / batch_size)

                # apply padding
                for i in range(len(batch_sentences)):
                    sent = batch_sentences[i]
                    candsyns = batch_candidates[i]

                    batch_sentences[i] = sent + [output_vocab["<PAD>"]] * (max_len - len(sent))
                    sent_pos = batch_pos[i]
                    batch_pos[i] = sent_pos + [pos_vocab["<PAD>"]] * (max_len - len(sent))
                    sent_coarse = batch_coarse[i]
                    batch_coarse[i] = sent_coarse + [coarse_vocab["<PAD>"]] * (max_len - len(sent))
                    batch_candidates[i] = candsyns + [pad_candidates] * (max_len - len(sent))

                    if batch_labels is not None:
                        lbls = batch_labels[i]
                        batch_labels[i] = lbls + [output_vocab["<PAD>"]] * (max_len - len(sent))

                # shuffle if needed
                if to_shuffle:
                    if batch_labels is not None:
                        batch_sentences, \
                            batch_pos, \
                            batch_coarse, \
                            batch_labels, \
                            batch_domains, \
                            batch_candidates, \
                            batch_lengths = shuffle(batch_sentences,
                                                    batch_pos,
                                                    batch_coarse,
                                                    batch_labels,
                                                    batch_domains,
                                                    batch_candidates,
                                                    batch_lengths)
                    else:
                        batch_sentences, \
                            batch_pos, \
                            batch_coarse, \
                            batch_domains, \
                            batch_candidates, \
                            batch_lengths = shuffle(batch_sentences,
                                                    batch_pos,
                                                    batch_coarse,
                                                    batch_domains,
                                                    batch_candidates,
                                                    batch_lengths)

                if batch_labels is not None:
                    yield batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_labels, batch_lengths
                else:
                    yield batch_sentences, batch_pos, batch_coarse, batch_domains, batch_candidates, batch_lengths
