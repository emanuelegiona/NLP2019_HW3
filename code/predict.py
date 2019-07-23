import tensorflow as tf

import utils as u
import corpus_handler as ch
from model import MFSTagger, BasicTagger, MultitaskTagger, DomainAwareTagger


# --- global variables ---
MFS = False
DATASET = "senseval3"
MODEL_ID = "multi_pos_wnDom_semcor"     # best scoring model
BATCH_SIZE = 16
LEARNING_RATE = 0.10
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 32
LAYERS = 2
# --- --- ---


def read_test_set(test_set_path, batch_size):
    batches = []

    with ch.TrainParser(file=test_set_path) as parser:
        batch = []
        for curr_sentence in parser.next_sentence():
            batch.append(curr_sentence)

            if len(batch) == batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)
            del batch

    return batches


def MFS_predict(input_path, output_path, resources_path, toWnDomains=False, toLEX=False):
    """
    General predict function to be used inside predict_babelnet, predict_wordnet_domains, and predict_lexicographer for code cleanness.
    ATTENTION: toWnDomains and toLEX are mutually exclusive.

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param toWnDomains: True: the output file contains WordNet Domains; False: BabelNet IDs (default)
    :param toLEX: True: the output file contains lexnames; False: BabelNet IDs (default)
    :return: None
    """

    assert not (toWnDomains and toLEX), "toWnDomains and toLEX are mutually exclusive"

    with open(output_path, mode="w"):
        pass

    with open(output_path, mode="a") as output:
        u.log_message(None, "Reading mappings...")
        bn2wn, wn2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2wordnet.tsv"))

        # --- additional mappings, if needed ---
        if toWnDomains:
            bn2output, output2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2wndomains.tsv"))
        elif toLEX:
            bn2output, output2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2lexnames.tsv"))

        u.log_message(None, "Performing predictions...")
        for test_set_batch in read_test_set(test_set_path=input_path, batch_size=BATCH_SIZE):
            for curr_sentence in test_set_batch:
                for curr_word in curr_sentence:
                    if curr_word.id is not None and curr_word.has_instance:
                        sense = MFSTagger.predict_word(curr_word.lemma)

                        # map WN ID to BabelNet ID
                        sense = wn2bn[sense][0]

                        # map BabelNet ID to the chosen output inventory (if needed)
                        if toWnDomains or toLEX:
                            sense = bn2output.get(sense, ["factotum"] if toWnDomains else ["misc"])[0]  # handle possible missing mappings with bogus classes

                        # write to file
                        output.write("%s %s\n" % (curr_word.id, sense))
                        output.flush()

        output.flush()
        u.log_message(None, "Done.")


def general_predict(input_path, output_path, resources_path, toWnDomains=False, toLEX=False):
    """
    General predict function to be used inside predict_babelnet, predict_wordnet_domains, and predict_lexicographer for code cleanness.
    ATTENTION: toWnDomains and toLEX are mutually exclusive.

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param toWnDomains: True: the output file contains WordNet Domains; False: BabelNet IDs (default)
    :param toLEX: True: the output file contains lexnames; False: BabelNet IDs (default)
    :return: None
    """

    assert not (toWnDomains and toLEX), "toWnDomains and toLEX are mutually exclusive"

    # overwrite files if needed
    with open(output_path, mode="w"):
        pass

    with \
            tf.Session() as sess, \
            open(output_path, mode="a") as output:

        u.log_message(None, "Reading mappings...")
        bn2wn, wn2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2wordnet.tsv"))

        # --- additional mappings, if needed ---
        if toWnDomains:
            bn2output, output2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2wndomains.tsv"))
        elif toLEX:
            bn2output, output2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2lexnames.tsv"))

        # --- model-specific mappings (if any) ---
        bn2coarse, coarse2bn = u.read_mapping("%s/%s" % (resources_path, "babelnet2wndomains.tsv"))
        # --- --- ---

        u.log_message(None, "Reading vocabularies...")
        senses, rev_senses = u.read_vocab("%s/%s" % (resources_path, "semcor.fine_senses.txt"))
        inputs, rev_inputs, antivocab = u.read_vocab("%s/%s" % (resources_path, "semcor.input.vocab.txt"),
                                                     "%s/%s" % (resources_path, "semcor.input.anti.txt"))

        # --- model-specific vocabularies (if any) ---
        pos_vocab, rev_pos = u.read_vocab("%s/%s" % (resources_path, "semcor.POS.txt"))
        coarse_vocab, rev_coarse = u.read_vocab("%s/%s" % (resources_path, "semcor.wndomains.txt"))
        # --- --- ---

        output_vocab, rev_output_vocab = u.merge_vocabs(senses, rev_senses, inputs)

        u.log_message(None, "Creating model...")
        # --- BasicTagger ---
        #model = BasicTagger(learning_rate=LEARNING_RATE,
        #                    embedding_size=EMBEDDING_SIZE,
        #                    hidden_size=HIDDEN_SIZE,
        #                    num_layers=LAYERS,
        #                    input_size=len(inputs),
        #                    output_size=len(output_vocab))

        # --- MultitaskTagger POS ---
        #model = MultitaskTagger(learning_rate=LEARNING_RATE,
        #                        embedding_size=EMBEDDING_SIZE,
        #                        hidden_size=HIDDEN_SIZE,
        #                        num_layers=LAYERS,
        #                        input_size=len(inputs),
        #                        output_size=len(output_vocab),
        #                        pos_size=len(pos_vocab))

        # --- MultitaskTagger WordNet Domains or lexnames ---
        #model = MultitaskTagger(learning_rate=LEARNING_RATE,
        #                        embedding_size=EMBEDDING_SIZE,
        #                        hidden_size=HIDDEN_SIZE,
        #                        num_layers=LAYERS,
        #                        input_size=len(inputs),
        #                        output_size=len(output_vocab),
        #                        coarse_size=len(coarse_vocab))

        # --- MultitaskTagger POS + (WordNet Domains or lexnames) - best scoring model: POS + WordNet Domains
        model = MultitaskTagger(learning_rate=LEARNING_RATE,
                                embedding_size=EMBEDDING_SIZE,
                                hidden_size=HIDDEN_SIZE,
                                num_layers=LAYERS,
                                input_size=len(inputs),
                                pos_size=len(pos_vocab),
                                output_size=len(output_vocab),
                                coarse_size=len(coarse_vocab))

        # --- DomainAwareTagger non-swapping or swapping states ---
        #model = DomainAwareTagger(learning_rate=LEARNING_RATE,
        #                          embedding_size=EMBEDDING_SIZE,
        #                          hidden_size=HIDDEN_SIZE,
        #                          num_layers=LAYERS,
        #                          input_size=len(inputs),
        #                          pos_size=len(pos_vocab),
        #                          output_size=len(output_vocab),
        #                          coarse_size=len(coarse_vocab),
        #                          swap_states=True)

        u.log_message(None, "Loading model...")
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path="%s/models/%s/model.ckpt" % (resources_path, MODEL_ID))

        u.log_message(None, "Performing predictions...")
        test_set_batches = read_test_set(test_set_path=input_path, batch_size=BATCH_SIZE)

        for batch_sentence, \
            _, \
            _, \
            batch_candidates, \
            batch_lengths in model.batch_generator(batch_size=BATCH_SIZE,
                                                   training_file_path=input_path,
                                                   antivocab=antivocab,
                                                   output_vocab=output_vocab,
                                                   pos_vocab=pos_vocab,
                                                   coarse_vocab=coarse_vocab,
                                                   wn2bn=wn2bn,
                                                   bn2coarse=bn2coarse,
                                                   gold_file_path=None,
                                                   to_shuffle=False):

            # fetch the current batch of sentences
            curr_test_batch = test_set_batches[0]
            test_set_batches = test_set_batches[1:]

            preds = sess.run(model.dense_fine,
                             feed_dict={model.sentence: batch_sentence,
                                        model.seq_lengths: batch_lengths,
                                        model.keep_prob: 1.0,
                                        model.rec_keep_prob: 1.0})

            _, real_preds = model.compute_accuracy(batch_candidates, preds, labels=None, return_predictions=True)

            for prediction in real_preds:
                curr_sentence = curr_test_batch[0]
                curr_test_batch = curr_test_batch[1:]

                length = len(curr_sentence)

                # strip special tokens and padding
                prediction = prediction[1:length+1]

                for i in range(length):
                    # only check predictions for instance words
                    curr_word = curr_sentence[i]
                    if curr_word.id is not None and curr_word.has_instance:
                        # check whether the prediction is a fine-grained sense or not
                        if 4 < prediction[i] < len(senses):  # ignore the 5 special tokens starting from 0, and the input vocabulary as well
                            sense = rev_senses[prediction[i]]
                        else:
                            # MFS fallback
                            sense = MFSTagger.predict_word(curr_word.lemma)

                        # map WN ID to BabelNet ID
                        sense = wn2bn[sense][0]

                        # map BabelNet ID to the chosen output inventory (if needed)
                        if toWnDomains or toLEX:
                            sense = bn2output.get(sense, ["factotum"] if toWnDomains else ["misc"])[0]  # handle possible missing mappings with bogus classes

                        # write to file
                        output.write("%s %s\n" % (curr_word.id, sense))
                        output.flush()

        output.flush()
        u.log_message(None, "Done.")


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    general_predict(input_path, output_path, resources_path)


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    general_predict(input_path, output_path, resources_path, toWnDomains=True)


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    general_predict(input_path, output_path, resources_path, toLEX=True)


if __name__ == "__main__":
    if MFS:
        MFS_predict(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/mfs.bn.txt" % DATASET,
            resources_path="../resources")

        MFS_predict(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/mfs.wnDom.txt" % DATASET,
            resources_path="../resources",
            toWnDomains=True)

        MFS_predict(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/mfs.lex.txt" % DATASET,
            resources_path="../resources",
            toLEX=True)
    else:
        tf.reset_default_graph()
        predict_babelnet(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.bn.txt" % (DATASET, MODEL_ID),
            resources_path="../resources")

        tf.reset_default_graph()
        predict_wordnet_domains(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.wnDom.txt" % (DATASET, MODEL_ID),
            resources_path="../resources")

        tf.reset_default_graph()
        predict_lexicographer(
            input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.data.xml" % (DATASET, DATASET),
            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.lex.txt" % (DATASET, MODEL_ID),
            resources_path="../resources")
