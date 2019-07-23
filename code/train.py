import tensorflow as tf

import utils as u
from model import MFSTagger, BasicTagger, MultitaskTagger, DomainAwareTagger


# --- global variables ---
SAVE_FREQUENCY = 10
GRID_SEARCH_TRAINING = 50
GRID_SEARCH_DEV = 15
# --- --- ---


def add_summary(writer, name, value, global_step):
    """
    Utility function to track the model's progess in TensorBoard
    :param writer: tf.summary.FileWriter instance
    :param name: Value label to be shown in TensorBoard
    :param value: Value to append for the current step
    :param global_step: Current step for which the value has to be considered
    :return: None
    """

    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)


def train_basic_tagger(model_ID, model_path,
                       training_file_path, gold_training_file_path,
                       dev_file_path, gold_dev_file_path,
                       fine_senses_vocab_path, input_vocab_path, input_antivocab_path,
                       learning_rate, embedding_size, hidden_size, layers,
                       keep_prob, rec_keep_prob,
                       batch_size, epochs,
                       grid_search=False):
    """
    Training routine for a BasicTagger model
    :param model_ID: ID to be used when referring to this model
    :param model_path: Path to the root directory to be used when saving this model
    :param training_file_path: Path to a file in Raganato's data.xml format
    :param gold_training_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to training_file_path)
    :param dev_file_path: Path to a file in Raganato's data.xml format
    :param gold_dev_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to dev_file_path)
    :param fine_senses_vocab_path: Path to a vocabulary of fine-grained senses (as built by utils.make_output_vocab)
    :param input_vocab_path: Path to a vocabulary of input words (as built by utils.make_input_vocab)
    :param input_antivocab_path: Path to a vocabulary of subsampled words (as built by utils.make_input_vocab)
    :param learning_rate: Learning rate to be used during optmization
    :param embedding_size: Size of embeddings to be used for the input words
    :param hidden_size: Size of the hidden layer of the LSTM layers
    :param layers: Number of LSTM layers to be stacked
    :param keep_prob: Probability of the input LSTM layer to keep the input
    :param rec_keep_prob: Probability of the recurrent layers in the LSTM to keep the input
    :param batch_size: Size of batches to be used during training
    :param epochs: Number of epochs the training has to last for
    :param grid_search: True: grid search training mode enabled, performs shorter training; False: normal training (default)
    :return: None
    """

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        u.log_message(log, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab(fine_senses_vocab_path)
        inputs, rev_inputs, antivocab = u.read_vocab(input_vocab_path, input_antivocab_path)

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)
        del senses, rev_senses, rev_inputs

        u.log_message(log, "Creating model...")
        model = BasicTagger(learning_rate=learning_rate,
                            embedding_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=layers,
                            input_size=len(inputs),
                            output_size=len(output_vocab))

        u.log_message(log, "\tModel ID: %s" % model_ID, with_time=False)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID), with_time=False)
        u.log_message(log, "\tLearning rate: %.3f" % learning_rate, with_time=False)
        u.log_message(log, "\tEmbedding size: %d" % embedding_size, with_time=False)
        u.log_message(log, "\tHidden size: %d" % hidden_size, with_time=False)
        u.log_message(log, "\tLayers: %d" % layers, with_time=False)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs + 1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            # training
            for batch_sentence, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=training_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       gold_file_path=gold_training_file_path):

                _, preds, loss_val = sess.run([model.train, model.dense_fine, model.loss],
                                              feed_dict={model.sentence: batch_sentence,
                                                         model.labels: batch_labels,
                                                         model.seq_lengths: batch_lengths,
                                                         model.keep_prob: keep_prob,
                                                         model.rec_keep_prob: rec_keep_prob})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter training during grid search
                if grid_search and iterations == GRID_SEARCH_TRAINING:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            train_loss = accumulated_loss
            train_acc = accumulated_acc

            u.log_message(log, "\tTrain loss: %.5f" % train_loss, with_time=False)
            add_summary(tf_logger,
                        "train_loss",
                        train_loss,
                        epoch)
            u.log_message(log, "\tTrain accuracy: %.5f" % train_acc, with_time=False)
            add_summary(tf_logger,
                        "train_acc",
                        train_acc,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_sentence, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=dev_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       gold_file_path=gold_dev_file_path,
                                                       to_shuffle=False):

                preds, loss_val = sess.run([model.dense_fine, model.loss],
                                           feed_dict={model.sentence: batch_sentence,
                                                      model.labels: batch_labels,
                                                      model.seq_lengths: batch_lengths,
                                                      model.keep_prob: 1.0,
                                                      model.rec_keep_prob: 1.0})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter evaluation as well during grid search
                if grid_search and iterations == GRID_SEARCH_DEV:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            dev_loss = accumulated_loss
            dev_acc = accumulated_acc

            u.log_message(log, "\tDev loss: %.5f" % dev_loss, with_time=False)
            add_summary(tf_logger,
                        "dev_loss",
                        dev_loss,
                        epoch)
            u.log_message(log, "\tDev accuracy: %.5f" % dev_acc, with_time=False)
            add_summary(tf_logger,
                        "dev_acc",
                        dev_acc,
                        epoch)

            if not grid_search and epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))


def train_multi_pos(model_ID, model_path,
                    training_file_path, gold_training_file_path,
                    dev_file_path, gold_dev_file_path,
                    fine_senses_vocab_path, input_vocab_path, input_antivocab_path,
                    pos_vocab_path,
                    learning_rate, embedding_size, hidden_size, layers,
                    keep_prob, rec_keep_prob,
                    batch_size, epochs,
                    grid_search=False):
    """
    Training routine for a MultitaskTagger model exploiting POS tags.
    :param model_ID: ID to be used when referring to this model
    :param model_path: Path to the root directory to be used when saving this model
    :param training_file_path: Path to a file in Raganato's data.xml format
    :param gold_training_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to training_file_path)
    :param dev_file_path: Path to a file in Raganato's data.xml format
    :param gold_dev_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to dev_file_path)
    :param fine_senses_vocab_path: Path to a vocabulary of fine-grained senses (as built by utils.make_output_vocab)
    :param pos_vocab_path: Path to a vocabulary of POS tags (as built by utils.make_POS_vocab)
    :param input_vocab_path: Path to a vocabulary of input words (as built by utils.make_input_vocab)
    :param input_antivocab_path: Path to a vocabulary of subsampled words (as built by utils.make_input_vocab)
    :param learning_rate: Learning rate to be used during optmization
    :param embedding_size: Size of embeddings to be used for the input words
    :param hidden_size: Size of the hidden layer of the LSTM layers
    :param layers: Number of LSTM layers to be stacked
    :param keep_prob: Probability of the input LSTM layer to keep the input
    :param rec_keep_prob: Probability of the recurrent layers in the LSTM to keep the input
    :param batch_size: Size of batches to be used during training
    :param epochs: Number of epochs the training has to last for
    :param grid_search: True: grid search training mode enabled, performs shorter training; False: normal training (default)
    :return: None
    """

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        u.log_message(log, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab(fine_senses_vocab_path)
        pos_vocab, rev_pos = u.read_vocab(pos_vocab_path)
        inputs, rev_inputs, antivocab = u.read_vocab(input_vocab_path, input_antivocab_path)

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)
        del senses, rev_senses, rev_inputs

        u.log_message(log, "Creating model...")
        model = MultitaskTagger(learning_rate=learning_rate,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=layers,
                                input_size=len(inputs),
                                output_size=len(output_vocab),
                                pos_size=len(pos_vocab))

        u.log_message(log, "\tModel ID: %s" % model_ID, with_time=False)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID), with_time=False)
        u.log_message(log, "\tLearning rate: %.3f" % learning_rate, with_time=False)
        u.log_message(log, "\tEmbedding size: %d" % embedding_size, with_time=False)
        u.log_message(log, "\tHidden size: %d" % hidden_size, with_time=False)
        u.log_message(log, "\tLayers: %d" % layers, with_time=False)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs + 1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            # training
            for batch_sentence, \
                batch_pos, \
                _, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=training_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       gold_file_path=gold_training_file_path):

                _, preds, loss_val = sess.run([model.train, model.dense_fine, model.loss],
                                              feed_dict={model.sentence: batch_sentence,
                                                         model.labels: batch_labels,
                                                         model.labels_pos: batch_pos,
                                                         model.seq_lengths: batch_lengths,
                                                         model.keep_prob: keep_prob,
                                                         model.rec_keep_prob: rec_keep_prob})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter training during grid search
                if grid_search and iterations == GRID_SEARCH_TRAINING:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            train_loss = accumulated_loss
            train_acc = accumulated_acc

            u.log_message(log, "\tTrain loss: %.5f" % train_loss, with_time=False)
            add_summary(tf_logger,
                        "train_loss",
                        train_loss,
                        epoch)
            u.log_message(log, "\tTrain accuracy: %.5f" % train_acc, with_time=False)
            add_summary(tf_logger,
                        "train_acc",
                        train_acc,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_sentence, \
                batch_pos, \
                _, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=dev_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       gold_file_path=gold_dev_file_path,
                                                       to_shuffle=False):

                preds, loss_val = sess.run([model.dense_fine, model.loss],
                                           feed_dict={model.sentence: batch_sentence,
                                                      model.labels: batch_labels,
                                                      model.labels_pos: batch_pos,
                                                      model.seq_lengths: batch_lengths,
                                                      model.keep_prob: 1.0,
                                                      model.rec_keep_prob: 1.0})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter evaluation as well during grid search
                if grid_search and iterations == GRID_SEARCH_DEV:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            dev_loss = accumulated_loss
            dev_acc = accumulated_acc

            u.log_message(log, "\tDev loss: %.5f" % dev_loss, with_time=False)
            add_summary(tf_logger,
                        "dev_loss",
                        dev_loss,
                        epoch)
            u.log_message(log, "\tDev accuracy: %.5f" % dev_acc, with_time=False)
            add_summary(tf_logger,
                        "dev_acc",
                        dev_acc,
                        epoch)

            if not grid_search and epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))


def train_multi_coarse(model_ID, model_path,
                       training_file_path, gold_training_file_path,
                       dev_file_path, gold_dev_file_path,
                       fine_senses_vocab_path, input_vocab_path, input_antivocab_path,
                       coarse_vocab_path, bn2wn_path, bn2coarse_path,
                       learning_rate, embedding_size, hidden_size, layers,
                       keep_prob, rec_keep_prob,
                       batch_size, epochs,
                       grid_search=False):
    """
    Training routine for a MultitaskTagger model exploiting coarse-grained senses.
    :param model_ID: ID to be used when referring to this model
    :param model_path: Path to the root directory to be used when saving this model
    :param training_file_path: Path to a file in Raganato's data.xml format
    :param gold_training_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to training_file_path)
    :param dev_file_path: Path to a file in Raganato's data.xml format
    :param gold_dev_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to dev_file_path)
    :param fine_senses_vocab_path: Path to a vocabulary of fine-grained senses (as built by utils.make_output_vocab)
    :param coarse_vocab_path: Path to a vocabulary of coarse-grained senses (as built by utils.make_output_vocab)
    :param bn2wn_path: Path to file containing the mapping from BabelNet IDs to WordNet IDs
    :param bn2coarse_path: Path to file containing the mapping from BabelNet IDs to the coarse-grained sense vocabulary of choice
    :param input_vocab_path: Path to a vocabulary of input words (as built by utils.make_input_vocab)
    :param input_antivocab_path: Path to a vocabulary of subsampled words (as built by utils.make_input_vocab)
    :param learning_rate: Learning rate to be used during optmization
    :param embedding_size: Size of embeddings to be used for the input words
    :param hidden_size: Size of the hidden layer of the LSTM layers
    :param layers: Number of LSTM layers to be stacked
    :param keep_prob: Probability of the input LSTM layer to keep the input
    :param rec_keep_prob: Probability of the recurrent layers in the LSTM to keep the input
    :param batch_size: Size of batches to be used during training
    :param epochs: Number of epochs the training has to last for
    :param grid_search: True: grid search training mode enabled, performs shorter training; False: normal training (default)
    :return: None
    """

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        u.log_message(log, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab(fine_senses_vocab_path)
        coarse_vocab, rev_coarse = u.read_vocab(coarse_vocab_path)
        inputs, rev_inputs, antivocab = u.read_vocab(input_vocab_path, input_antivocab_path)

        u.log_message(log, "Reading mappings...")
        bn2wn, wn2bn = u.read_mapping(bn2wn_path)
        bn2coarse, coarse2bn = u.read_mapping(bn2coarse_path)

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)
        del senses, rev_senses, rev_inputs

        u.log_message(log, "Creating model...")
        model = MultitaskTagger(learning_rate=learning_rate,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=layers,
                                input_size=len(inputs),
                                output_size=len(output_vocab),
                                coarse_size=len(coarse_vocab))

        u.log_message(log, "\tModel ID: %s" % model_ID, with_time=False)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID), with_time=False)
        u.log_message(log, "\tLearning rate: %.3f" % learning_rate, with_time=False)
        u.log_message(log, "\tEmbedding size: %d" % embedding_size, with_time=False)
        u.log_message(log, "\tHidden size: %d" % hidden_size, with_time=False)
        u.log_message(log, "\tLayers: %d" % layers, with_time=False)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs + 1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            # training
            for batch_sentence, \
                _, \
                batch_coarse, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=training_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_training_file_path):

                _, preds, loss_val = sess.run([model.train, model.dense_fine, model.loss],
                                              feed_dict={model.sentence: batch_sentence,
                                                         model.labels: batch_labels,
                                                         model.labels_coarse: batch_coarse,
                                                         model.seq_lengths: batch_lengths,
                                                         model.keep_prob: keep_prob,
                                                         model.rec_keep_prob: rec_keep_prob})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter training during grid search
                if grid_search and iterations == GRID_SEARCH_TRAINING:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            train_loss = accumulated_loss
            train_acc = accumulated_acc

            u.log_message(log, "\tTrain loss: %.5f" % train_loss, with_time=False)
            add_summary(tf_logger,
                        "train_loss",
                        train_loss,
                        epoch)
            u.log_message(log, "\tTrain accuracy: %.5f" % train_acc, with_time=False)
            add_summary(tf_logger,
                        "train_acc",
                        train_acc,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_sentence, \
                _, \
                batch_coarse, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=dev_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_dev_file_path,
                                                       to_shuffle=False):

                preds, loss_val = sess.run([model.dense_fine, model.loss],
                                           feed_dict={model.sentence: batch_sentence,
                                                      model.labels: batch_labels,
                                                      model.labels_coarse: batch_coarse,
                                                      model.seq_lengths: batch_lengths,
                                                      model.keep_prob: 1.0,
                                                      model.rec_keep_prob: 1.0})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter evaluation as well during grid search
                if grid_search and iterations == GRID_SEARCH_DEV:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            dev_loss = accumulated_loss
            dev_acc = accumulated_acc

            u.log_message(log, "\tDev loss: %.5f" % dev_loss, with_time=False)
            add_summary(tf_logger,
                        "dev_loss",
                        dev_loss,
                        epoch)
            u.log_message(log, "\tDev accuracy: %.5f" % dev_acc, with_time=False)
            add_summary(tf_logger,
                        "dev_acc",
                        dev_acc,
                        epoch)

            if not grid_search and epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))


def train_multi_combined(model_ID, model_path,
                         training_file_path, gold_training_file_path,
                         dev_file_path, gold_dev_file_path,
                         fine_senses_vocab_path, input_vocab_path, input_antivocab_path,
                         pos_vocab_path, coarse_vocab_path, bn2wn_path, bn2coarse_path,
                         learning_rate, embedding_size, hidden_size, layers,
                         keep_prob, rec_keep_prob,
                         batch_size, epochs,
                         grid_search=False):
    """
    Training routine for a MultitaskTagger model exploiting both POS tags and coarse-grained senses.
    :param model_ID: ID to be used when referring to this model
    :param model_path: Path to the root directory to be used when saving this model
    :param training_file_path: Path to a file in Raganato's data.xml format
    :param gold_training_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to training_file_path)
    :param dev_file_path: Path to a file in Raganato's data.xml format
    :param gold_dev_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to dev_file_path)
    :param fine_senses_vocab_path: Path to a vocabulary of fine-grained senses (as built by utils.make_output_vocab)
    :param pos_vocab_path: Path to a vocabulary of POS tags (as built by utils.make_POS_vocab)
    :param coarse_vocab_path: Path to a vocabulary of coarse-grained senses (as built by utils.make_output_vocab)
    :param bn2wn_path: Path to file containing the mapping from BabelNet IDs to WordNet IDs
    :param bn2coarse_path: Path to file containing the mapping from BabelNet IDs to the coarse-grained sense vocabulary of choice
    :param input_vocab_path: Path to a vocabulary of input words (as built by utils.make_input_vocab)
    :param input_antivocab_path: Path to a vocabulary of subsampled words (as built by utils.make_input_vocab)
    :param learning_rate: Learning rate to be used during optmization
    :param embedding_size: Size of embeddings to be used for the input words
    :param hidden_size: Size of the hidden layer of the LSTM layers
    :param layers: Number of LSTM layers to be stacked
    :param keep_prob: Probability of the input LSTM layer to keep the input
    :param rec_keep_prob: Probability of the recurrent layers in the LSTM to keep the input
    :param batch_size: Size of batches to be used during training
    :param epochs: Number of epochs the training has to last for
    :param grid_search: True: grid search training mode enabled, performs shorter training; False: normal training (default)
    :return: None
    """

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        u.log_message(log, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab(fine_senses_vocab_path)
        pos_vocab, rev_pos = u.read_vocab(pos_vocab_path)
        coarse_vocab, rev_coarse = u.read_vocab(coarse_vocab_path)
        inputs, rev_inputs, antivocab = u.read_vocab(input_vocab_path, input_antivocab_path)

        u.log_message(log, "Reading mappings...")
        bn2wn, wn2bn = u.read_mapping(bn2wn_path)
        bn2coarse, coarse2bn = u.read_mapping(bn2coarse_path)

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)
        del senses, rev_senses, rev_inputs

        u.log_message(log, "Creating model...")
        model = MultitaskTagger(learning_rate=learning_rate,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=layers,
                                input_size=len(inputs),
                                pos_size=len(pos_vocab),
                                output_size=len(output_vocab),
                                coarse_size=len(coarse_vocab))

        u.log_message(log, "\tModel ID: %s" % model_ID, with_time=False)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID), with_time=False)
        u.log_message(log, "\tLearning rate: %.3f" % learning_rate, with_time=False)
        u.log_message(log, "\tEmbedding size: %d" % embedding_size, with_time=False)
        u.log_message(log, "\tHidden size: %d" % hidden_size, with_time=False)
        u.log_message(log, "\tLayers: %d" % layers, with_time=False)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs + 1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            # training
            for batch_sentence, \
                batch_pos, \
                batch_coarse, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=training_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_training_file_path):

                _, preds, loss_val = sess.run([model.train, model.dense_fine, model.loss],
                                              feed_dict={model.sentence: batch_sentence,
                                                         model.labels: batch_labels,
                                                         model.labels_pos: batch_pos,
                                                         model.labels_coarse: batch_coarse,
                                                         model.seq_lengths: batch_lengths,
                                                         model.keep_prob: keep_prob,
                                                         model.rec_keep_prob: rec_keep_prob})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter training during grid search
                if grid_search and iterations == GRID_SEARCH_TRAINING:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            train_loss = accumulated_loss
            train_acc = accumulated_acc

            u.log_message(log, "\tTrain loss: %.5f" % train_loss, with_time=False)
            add_summary(tf_logger,
                        "train_loss",
                        train_loss,
                        epoch)
            u.log_message(log, "\tTrain accuracy: %.5f" % train_acc, with_time=False)
            add_summary(tf_logger,
                        "train_acc",
                        train_acc,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_sentence, \
                batch_pos, \
                batch_coarse, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=dev_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_dev_file_path,
                                                       to_shuffle=False):

                preds, loss_val = sess.run([model.dense_fine, model.loss],
                                           feed_dict={model.sentence: batch_sentence,
                                                      model.labels: batch_labels,
                                                      model.labels_pos: batch_pos,
                                                      model.labels_coarse: batch_coarse,
                                                      model.seq_lengths: batch_lengths,
                                                      model.keep_prob: 1.0,
                                                      model.rec_keep_prob: 1.0})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter evaluation as well during grid search
                if grid_search and iterations == GRID_SEARCH_DEV:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            dev_loss = accumulated_loss
            dev_acc = accumulated_acc

            u.log_message(log, "\tDev loss: %.5f" % dev_loss, with_time=False)
            add_summary(tf_logger,
                        "dev_loss",
                        dev_loss,
                        epoch)
            u.log_message(log, "\tDev accuracy: %.5f" % dev_acc, with_time=False)
            add_summary(tf_logger,
                        "dev_acc",
                        dev_acc,
                        epoch)

            if not grid_search and epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))


def train_domain_aware(model_ID, model_path,
                       training_file_path, gold_training_file_path,
                       dev_file_path, gold_dev_file_path,
                       fine_senses_vocab_path, input_vocab_path, input_antivocab_path,
                       pos_vocab_path, coarse_vocab_path, bn2wn_path, bn2coarse_path,
                       swap_states,
                       learning_rate, embedding_size, hidden_size, layers,
                       keep_prob, rec_keep_prob,
                       batch_size, epochs,
                       grid_search=False):
    """
    Training routine for a DomainAwareTagger model.
    :param model_ID: ID to be used when referring to this model
    :param model_path: Path to the root directory to be used when saving this model
    :param training_file_path: Path to a file in Raganato's data.xml format
    :param gold_training_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to training_file_path)
    :param dev_file_path: Path to a file in Raganato's data.xml format
    :param gold_dev_file_path: Path to a gold file in Raganato's gold.key.txt format (associated to dev_file_path)
    :param fine_senses_vocab_path: Path to a vocabulary of fine-grained senses (as built by utils.make_output_vocab)
    :param pos_vocab_path: Path to a vocabulary of POS tags (as built by utils.make_POS_vocab)
    :param coarse_vocab_path: Path to a vocabulary of coarse-grained senses (as built by utils.make_output_vocab)
    :param bn2wn_path: Path to file containing the mapping from BabelNet IDs to WordNet IDs
    :param bn2coarse_path: Path to file containing the mapping from BabelNet IDs to the coarse-grained sense vocabulary of choice
    :param swap_states: if True, biLSTM hidden states are inverted in-between stacked layers
    :param input_vocab_path: Path to a vocabulary of input words (as built by utils.make_input_vocab)
    :param input_antivocab_path: Path to a vocabulary of subsampled words (as built by utils.make_input_vocab)
    :param learning_rate: Learning rate to be used during optmization
    :param embedding_size: Size of embeddings to be used for the input words
    :param hidden_size: Size of the hidden layer of the LSTM layers
    :param layers: Number of LSTM layers to be stacked
    :param keep_prob: Probability of the input LSTM layer to keep the input
    :param rec_keep_prob: Probability of the recurrent layers in the LSTM to keep the input
    :param batch_size: Size of batches to be used during training
    :param epochs: Number of epochs the training has to last for
    :param grid_search: True: grid search training mode enabled, performs shorter training; False: normal training (default)
    :return: None
    """

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        u.log_message(log, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab(fine_senses_vocab_path)
        pos_vocab, rev_pos = u.read_vocab(pos_vocab_path)
        coarse_vocab, rev_coarse = u.read_vocab(coarse_vocab_path)
        inputs, rev_inputs, antivocab = u.read_vocab(input_vocab_path, input_antivocab_path)

        u.log_message(log, "Reading mappings...")
        bn2wn, wn2bn = u.read_mapping(bn2wn_path)
        bn2coarse, coarse2bn = u.read_mapping(bn2coarse_path)

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)
        del senses, rev_senses, rev_inputs

        u.log_message(log, "Creating model...")
        model = DomainAwareTagger(learning_rate=learning_rate,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=layers,
                                  input_size=len(inputs),
                                  pos_size=len(pos_vocab),
                                  output_size=len(output_vocab),
                                  coarse_size=len(coarse_vocab),
                                  swap_states=swap_states)

        u.log_message(log, "\tModel ID: %s" % model_ID, with_time=False)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID), with_time=False)
        u.log_message(log, "\tLearning rate: %.3f" % learning_rate, with_time=False)
        u.log_message(log, "\tEmbedding size: %d" % embedding_size, with_time=False)
        u.log_message(log, "\tHidden size: %d" % hidden_size, with_time=False)
        u.log_message(log, "\tLayers: %d" % layers, with_time=False)
        u.log_message(log, "\tSwap states: %s" % ("Yes" if swap_states else "No"), with_time=False)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs + 1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            # training
            for batch_sentence, \
                batch_pos, \
                batch_coarse, \
                batch_domains, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=training_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_training_file_path):

                _, preds, loss_val = sess.run([model.train, model.dense_fine, model.loss],
                                              feed_dict={model.sentence: batch_sentence,
                                                         model.labels: batch_labels,
                                                         model.labels_pos: batch_pos,
                                                         model.labels_coarse: batch_coarse,
                                                         model.label_domain: batch_domains,
                                                         model.seq_lengths: batch_lengths,
                                                         model.keep_prob: keep_prob,
                                                         model.rec_keep_prob: rec_keep_prob})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter training during grid search
                if grid_search and iterations == GRID_SEARCH_TRAINING:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            train_loss = accumulated_loss
            train_acc = accumulated_acc

            u.log_message(log, "\tTrain loss: %.5f" % train_loss, with_time=False)
            add_summary(tf_logger,
                        "train_loss",
                        train_loss,
                        epoch)
            u.log_message(log, "\tTrain accuracy: %.5f" % train_acc, with_time=False)
            add_summary(tf_logger,
                        "train_acc",
                        train_acc,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_sentence, \
                batch_pos, \
                batch_coarse, \
                batch_domains, \
                batch_candidates, \
                batch_labels, \
                batch_lengths in model.batch_generator(batch_size=batch_size,
                                                       training_file_path=dev_file_path,
                                                       antivocab=antivocab,
                                                       output_vocab=output_vocab,
                                                       pos_vocab=pos_vocab,
                                                       coarse_vocab=coarse_vocab,
                                                       wn2bn=wn2bn,
                                                       bn2coarse=bn2coarse,
                                                       gold_file_path=gold_dev_file_path,
                                                       to_shuffle=False):

                preds, loss_val = sess.run([model.dense_fine, model.loss],
                                           feed_dict={model.sentence: batch_sentence,
                                                      model.labels: batch_labels,
                                                      model.labels_pos: batch_pos,
                                                      model.labels_coarse: batch_coarse,
                                                      model.label_domain: batch_domains,
                                                      model.seq_lengths: batch_lengths,
                                                      model.keep_prob: 1.0,
                                                      model.rec_keep_prob: 1.0})

                accumulated_loss += loss_val
                accumulated_acc += model.compute_accuracy(batch_candidates, preds, batch_labels)
                iterations += 1

                # shorter evaluation as well during grid search
                if grid_search and iterations == GRID_SEARCH_DEV:
                    break

            accumulated_loss /= iterations
            accumulated_acc /= iterations
            dev_loss = accumulated_loss
            dev_acc = accumulated_acc

            u.log_message(log, "\tDev loss: %.5f" % dev_loss, with_time=False)
            add_summary(tf_logger,
                        "dev_loss",
                        dev_loss,
                        epoch)
            u.log_message(log, "\tDev accuracy: %.5f" % dev_acc, with_time=False)
            add_summary(tf_logger,
                        "dev_acc",
                        dev_acc,
                        epoch)

            if not grid_search and epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))


def grid_search():
    """
    Utility function to perform hyper-parameters tuning.
    :return: None
    """

    embedding_sizes = [32, 64]
    hidden_sizes = [32, 64]
    learning_rates = [0.15, 0.10]
    n_layers = [1, 2]

    for e_size in embedding_sizes:
        for h_size in hidden_sizes:
            for lr in learning_rates:
                for nl in n_layers:
                    tf.reset_default_graph()

                    train_basic_tagger(model_ID="basic_layers%d_emb%d_hid%d_lr%.3f" % (nl, e_size, h_size, lr),
                                       model_path="../resources/models/",
                                       training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
                                       gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
                                       dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
                                       gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
                                       fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
                                       input_vocab_path="../resources/semcor.input.vocab.txt",
                                       input_antivocab_path="../resources/semcor.input.anti.txt",
                                       learning_rate=lr,
                                       embedding_size=e_size,
                                       hidden_size=h_size,
                                       layers=nl,
                                       keep_prob=0.75,
                                       rec_keep_prob=0.8,
                                       batch_size=16,
                                       epochs=20,
                                       grid_search=True)


if __name__ == "__main__":
    #grid_search()
    # best tuning:
    # - embedding size: 64
    # - hidden size: 32
    # - layers: 2
    # - learning rate: 0.10
    # - train: 85.65%; dev: 91.12% (NOT from Raganato's scorer)

    # --- BasicTagger ---
    #train_basic_tagger(model_ID="basictagger_semcor",
    #                   model_path="../resources/models/",
    #                   training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                   gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                   dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                   gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                   fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                   input_vocab_path="../resources/semcor.input.vocab.txt",
    #                   input_antivocab_path="../resources/semcor.input.anti.txt",
    #                   learning_rate=0.10,
    #                   embedding_size=64,
    #                   hidden_size=32,
    #                   layers=2,
    #                   keep_prob=0.75,
    #                   rec_keep_prob=0.8,
    #                   batch_size=16,
    #                   epochs=100)

    # --- MultitaskTagger + POS ---
    #train_multi_pos(model_ID="multi_pos_semcor",
    #                model_path="../resources/models/",
    #                training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                pos_vocab_path="../resources/semcor.POS.txt",
    #                input_vocab_path="../resources/semcor.input.vocab.txt",
    #                input_antivocab_path="../resources/semcor.input.anti.txt",
    #                learning_rate=0.10,
    #                embedding_size=64,
    #                hidden_size=32,
    #                layers=2,
    #                keep_prob=0.75,
    #                rec_keep_prob=0.8,
    #                batch_size=16,
    #                epochs=100)

    # --- MultitaskTagger + WordNet Domains ---
    #train_multi_coarse(model_ID="multi_wnDom_semcor",
    #                   model_path="../resources/models/",
    #                   training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                   gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                   dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                   gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                   fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                   coarse_vocab_path="../resources/semcor.wndomains.txt",
    #                   bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                   bn2coarse_path="../resources/babelnet2wndomains.tsv",
    #                   input_vocab_path="../resources/semcor.input.vocab.txt",
    #                   input_antivocab_path="../resources/semcor.input.anti.txt",
    #                   learning_rate=0.10,
    #                   embedding_size=64,
    #                   hidden_size=32,
    #                   layers=2,
    #                   keep_prob=0.75,
    #                   rec_keep_prob=0.8,
    #                   batch_size=16,
    #                   epochs=100)

    # --- MultitaskTagger + Lexnames ---
    #train_multi_coarse(model_ID="multi_LEX_semcor",
    #                   model_path="../resources/models/",
    #                   training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                   gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                   dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                   gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                   fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                   coarse_vocab_path="../resources/semcor.lexnames.txt",
    #                   bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                   bn2coarse_path="../resources/babelnet2lexnames.tsv",
    #                   input_vocab_path="../resources/semcor.input.vocab.txt",
    #                   input_antivocab_path="../resources/semcor.input.anti.txt",
    #                   learning_rate=0.10,
    #                   embedding_size=64,
    #                   hidden_size=32,
    #                   layers=2,
    #                   keep_prob=0.75,
    #                   rec_keep_prob=0.8,
    #                   batch_size=16,
    #                   epochs=100)

    # --- MultitaskTagger + POS + WordNet Domains ---
    #train_multi_combined(model_ID="multi_pos_wnDom_semcor",
    #                     model_path="../resources/models/",
    #                     training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                     gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                     dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                     gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                     fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                     pos_vocab_path="../resources/semcor.POS.txt",
    #                     coarse_vocab_path="../resources/semcor.wndomains.txt",
    #                     bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                     bn2coarse_path="../resources/babelnet2wndomains.tsv",
    #                     input_vocab_path="../resources/semcor.input.vocab.txt",
    #                     input_antivocab_path="../resources/semcor.input.anti.txt",
    #                     learning_rate=0.10,
    #                     embedding_size=64,
    #                     hidden_size=32,
    #                     layers=2,
    #                     keep_prob=0.75,
    #                     rec_keep_prob=0.8,
    #                     batch_size=16,
    #                     epochs=100)

    # --- MultitaskTagger + POS + Lexnames ---
    #train_multi_combined(model_ID="multi_pos_LEX_semcor",
    #                     model_path="../resources/models/",
    #                     training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                     gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                     dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                     gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                     fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                     pos_vocab_path="../resources/semcor.POS.txt",
    #                     coarse_vocab_path="../resources/semcor.lexnames.txt",
    #                     bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                     bn2coarse_path="../resources/babelnet2lexnames.tsv",
    #                     input_vocab_path="../resources/semcor.input.vocab.txt",
    #                     input_antivocab_path="../resources/semcor.input.anti.txt",
    #                     learning_rate=0.10,
    #                     embedding_size=64,
    #                     hidden_size=32,
    #                     layers=2,
    #                     keep_prob=0.75,
    #                     rec_keep_prob=0.8,
    #                     batch_size=16,
    #                     epochs=100)

    # --- DomainAwareTagger (no swapping states) ---
    #train_domain_aware(model_ID="domain_noswap_semcor",
    #                   model_path="../resources/models/",
    #                   training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                   gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                   dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                   gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                   fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                   pos_vocab_path="../resources/semcor.POS.txt",
    #                   coarse_vocab_path="../resources/semcor.wndomains.txt",
    #                   bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                   bn2coarse_path="../resources/babelnet2wndomains.tsv",
    #                   swap_states=False,
    #                   input_vocab_path="../resources/semcor.input.vocab.txt",
    #                   input_antivocab_path="../resources/semcor.input.anti.txt",
    #                   learning_rate=0.10,
    #                   embedding_size=64,
    #                   hidden_size=32,
    #                   layers=2,
    #                   keep_prob=0.75,
    #                   rec_keep_prob=0.8,
    #                   batch_size=16,
    #                   epochs=100)

    # --- DomainAwareTagger (swapping states) ---
    #train_domain_aware(model_ID="domain_swap_semcor",
    #                   model_path="../resources/models/",
    #                   training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #                   gold_training_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #                   dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
    #                   gold_dev_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                   fine_senses_vocab_path="../resources/semcor.fine_senses.txt",
    #                   pos_vocab_path="../resources/semcor.POS.txt",
    #                   coarse_vocab_path="../resources/semcor.wndomains.txt",
    #                   bn2wn_path="../resources/babelnet2wordnet.tsv",
    #                   bn2coarse_path="../resources/babelnet2wndomains.tsv",
    #                   swap_states=True,
    #                   input_vocab_path="../resources/semcor.input.vocab.txt",
    #                   input_antivocab_path="../resources/semcor.input.anti.txt",
    #                   learning_rate=0.10,
    #                   embedding_size=64,
    #                   hidden_size=32,
    #                   layers=2,
    #                   keep_prob=0.75,
    #                   rec_keep_prob=0.8,
    #                   batch_size=16,
    #                   epochs=100)

    pass
