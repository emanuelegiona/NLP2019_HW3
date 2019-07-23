from datetime import datetime
from nltk.corpus import wordnet as wn
import corpus_handler as ch
from math import sqrt
from random import uniform
import copy


# --- global variables ---
BATCH_SIZE_FILE = 128
# --- --- ---


def get_time():
    """
    Gets the current time
    :return: Current time, as String
    """

    return str(datetime.now())


def log_message(file_handle, message, to_stdout=True, with_time=True):
    """
    Log utility function
    :param file_handle: Open file handle to the log file
    :param message: Log message
    :param to_stdout: True: also prints the message to the standard output (default); False: only writes to file
    :param with_time: True: appends time at the end of the line (default); False: only prints the given message
    :return: None
    """

    if with_time:
        message = "%s [%s]" % (message, get_time())

    if file_handle is not None:
        file_handle.write("%s\n" % message)
        file_handle.flush()

    if to_stdout:
        print("%s" % message)


def wn_id_from_synset(synset):
    """
    Builds the WordNet ID in the shape of wn:<offset><pos> for the given synset.
    :param synset: Synset to get the ID from
    :return: WordNet ID as described
    """

    offset = str(synset.offset())
    offset = "0" * (8 - len(offset)) + offset  # append heading 0s to the offset
    wn_id = "wn:%s%s" % (offset, synset.pos())

    return wn_id


def wn_id_from_sense_key(sense_key):
    """
    Returns a WordNet ID built out of offset and POS from a given WordNet sense key.
    :param sense_key: WordNet sense key
    :return: WordNet ID corresponding to the given sense key
    """

    synset = wn.lemma_from_key(sense_key).synset()
    return wn_id_from_synset(synset)


def candidate_synsets(lemma, pos):
    """
    Retrieves the candidate synsets for the given lemma and pos combination.
    :param lemma: Lemma to get the synsets of
    :param pos: POS associated to the lemma
    :return: Candidate synsets having the given lemma and POS, as List; the lemma itself in case there is no match in WordNet
    """

    pos_dictionary = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB}   # open classes only
    if pos == "." or pos == "PUNCT":
        return ["<PUNCTUATION>"]
    elif pos == "NUM":
        return ["<NUMBER>"]
    elif pos == "SYM":
        return ["<SYMBOL>"]
    elif pos in pos_dictionary:
        synsets = wn.synsets(lemma, pos=pos_dictionary[pos])
    else:
        synsets = wn.synsets(lemma)

    if len(synsets) == 0:
        return [lemma]
    return [wn_id_from_synset(syn) for syn in synsets]


def read_mapping(file_path):
    """
    Reads mappings from a tab separated file.
    :param file_path: Path to the mapping file
    :return: (mapping as Dict str -> List of str, reverse_mapping as Dict str -> List of str)
    """

    mapping = {}
    reverse_mapping = {}

    with open(file_path, mode="r") as f:
        for line in f:
            line = line.strip().split("\t")

            if len(line) < 2:
                continue

            key = line[0]
            if key not in mapping:
                mapping[key] = []
                for value in line[1:]:
                    mapping[key].append(value)
            else:
                for value in line[1:]:
                    mapping[key].append(value)

            for value in line[1:]:
                if value not in reverse_mapping:
                    reverse_mapping[value] = [key]
                else:
                    reverse_mapping[value].append(key)

    return mapping, reverse_mapping


def make_input_vocab(train_file_path, vocab_path, antivocab_path, min_count=5, subsampling=1e-4, logfile=None):
    """
    Creates a vocabulary of the given size from the given training data.xml file, sorting words by frequency.
    :param train_file_path: File path to the data.xml file in Raganato's format
    :param vocab_path: File path where to export the vocabulary to
    :param antivocab_path: File path where to export subsampled words to, in order not to consider them during training
    :param min_count: Minimum number of occurrences for words to be considered (default: 5)
    :param subsampling: Subsampling factor for frequent words (default: 10e-3)
    :param logfile: File handle for logging purposes
    :return: None
    """

    vocab_batch = []
    occurrences = {}
    anti_vocab = set()
    untouchable = set()

    # Overwrite with fresh file, if already existing
    with open(vocab_path, mode='w', encoding='utf-8'):
        pass
    with open(antivocab_path, mode='w', encoding='utf-8'):
        pass

    with \
            open(vocab_path, mode='a', encoding='utf-8') as output, \
            open(antivocab_path, mode='a', encoding='utf-8') as antioutput, \
            ch.TrainParser(train_file_path) as parser:

        log_message(logfile, "Started vocabulary creation")
        log_message(logfile, "Started vocabulary indexing")

        count = 0
        total_words = 0
        for sentence in parser.next_sentence():
            count += 1

            for word in sentence:
                lemma = word.lemma

                if word.pos == "." or word.pos == "PUNCT":
                    lemma = "<PUNCTUATION>"
                elif word.pos == "NUM":
                    lemma = "<NUMBER>"
                elif word.pos == "SYM":
                    lemma = "<SYMBOL>"

                # never apply sub-sampling to instance words
                if word.id is not None and word.has_instance:
                    untouchable.add(lemma)

                occurrences[lemma] = occurrences.get(lemma, 0) + 1
                total_words += 1

            if count % 50_000 == 0:
                log_message(logfile, "%d sentences parsed; %d words so far" % (count, len(occurrences)))

        log_message(logfile, "Finished vocabulary indexing; %d total sentences; %d total words" % (count, len(occurrences)))

        # sort by decreasing occurrence number to only get the most frequent words
        for w in sorted(occurrences.items(), key=lambda x: x[1], reverse=True):
            # only consider words with at least min_count occurrences (or instance words)
            if w[0] in untouchable or w[1] >= min_count:

                # apply subsampling
                prob = 1.0 - sqrt(subsampling / w[1] * total_words)
                if w[0] in untouchable or uniform(0, 1) >= prob:
                    # take the word
                    vocab_batch.append(w[0])
                else:
                    anti_vocab.add(w[0])

            # BATCH_SIZE reached
            if len(vocab_batch) == BATCH_SIZE_FILE:
                output.writelines("%s\n" % v for v in vocab_batch)
                vocab_batch = []

        del occurrences

        # incomplete batch
        if len(vocab_batch) > 0:
            output.writelines("%s\n" % v for v in vocab_batch)
            del vocab_batch

        # export too frequent words
        antioutput.writelines("%s\n" % a for a in anti_vocab)

    log_message(logfile, "Finished vocabulary creation")


def make_POS_vocab(train_file_path, vocab_path):
    """
    Creates a vocabulary of the given size from the given training data.xml file, sorting words by frequency.
    :param train_file_path: File path to the data.xml file in Raganato's format
    :param vocab_path: File path where to export the vocabulary to
    :return: None
    """

    vocab_batch = []
    occurrences = {}

    # Overwrite with fresh file, if already existing
    with open(vocab_path, mode='w', encoding='utf-8'):
        pass

    with \
            open(vocab_path, mode='a', encoding='utf-8') as output, \
            ch.TrainParser(train_file_path) as parser:

        log_message(None, "Started vocabulary creation")
        log_message(None, "Started vocabulary indexing")

        count = 0
        total_words = 0
        for sentence in parser.next_sentence():
            count += 1

            for word in sentence:
                pos = word.pos
                occurrences[pos] = occurrences.get(pos, 0) + 1
                total_words += 1

            if count % 50_000 == 0:
                log_message(None, "%d sentences parsed; %d POS tags so far" % (count, len(occurrences)))

        log_message(None,
                    "Finished vocabulary indexing; %d total sentences; %d total POS tags" % (count, len(occurrences)))

        # sort by decreasing occurrence number to only get the most frequent words
        for w in sorted(occurrences.items(), key=lambda x: x[1], reverse=True):
            vocab_batch.append(w[0])

            # BATCH_SIZE reached
            if len(vocab_batch) == BATCH_SIZE_FILE:
                output.writelines("%s\n" % v for v in vocab_batch)
                vocab_batch = []

        del occurrences

        # incomplete batch
        if len(vocab_batch) > 0:
            output.writelines("%s\n" % v for v in vocab_batch)
            del vocab_batch

    log_message(None, "Finished vocabulary creation")


def make_output_vocab(gold_file_path, output_path, to_LEX=False, bn_LEX_path=None, to_wnDomains=False, bn_wnD_path=None, bn_wn_path=None):
    """
    Builds the output vocabulary given a gold.key.txt file compliant to the Raganato's format.
    ATTENTION: to_LEX and to_wnDomains are mutually exclusive, only one of the two can be True at once.

    :param gold_file_path: Path to the gold.key.txt file
    :param output_path: Path to the file to be written
    :param to_LEX: True: output vocabulary in terms of lexnames; False: WordNet IDs (default)
    :param bn_LEX_path: Path to the BabelNet to lexnames mapping; MUST be provided if to_LEX is True
    :param to_wnDomains: True: output vocabulary in terms of WordNet Domains; False: WordNet IDs (default)
    :param bn_wnD_path: Path to the BabelNet to WN Domains mapping; MUST be provided if to_wnDomains is True
    :param bn_wn_path: Path to the BabelNet to WordNet mapping; MUST be provided if either to_LEX or to_wnDomains is True
    :return: None
    """

    assert not (to_LEX and to_wnDomains), "to_LEX and to_wnDomains are mutually exclusive"
    assert (not to_LEX or (to_LEX and bn_LEX_path is not None)), "bn_LEX_path MUST be provided if to_LEX is True"
    assert (not to_wnDomains or (to_wnDomains and bn_wnD_path is not None)), "bn_wnD_path MUST be provided if to_wnDomains is True"
    assert (not (to_LEX or to_wnDomains) or ((to_LEX or to_wnDomains) and bn_wn_path is not None)), "bn_wn_path MUST be provided if either to_LEX or to_wnDomains is True"

    if to_LEX or to_wnDomains:
        print("Reading mappings...")
        bn_wn, wn_bn = read_mapping(bn_wn_path)

        mapping_path = None
        if to_LEX:
            mapping_path = bn_LEX_path
        elif to_wnDomains:
            mapping_path = bn_wnD_path

        if mapping_path is not None:
            mapping, rev_mapping = read_mapping(mapping_path)

    print("Parsing started...")
    vocabulary = set()
    with \
            open(gold_file_path, mode="r") as gold,\
            open(output_path, mode="w") as output:

        for line in gold:
            line = line.strip().split(" ")
            if len(line) < 2:
                continue

            for sense_key in line[1:]:
                wn_id = wn_id_from_sense_key(sense_key)
                out_strings = [wn_id]

                if to_LEX or to_wnDomains:
                    out_strings = []
                    for bn_id in wn_bn[wn_id]:
                        try:
                            values = mapping[bn_id]
                        except KeyError:
                            values = []

                        out_strings.extend(values)

                for out_str in out_strings:
                    if out_str not in vocabulary:
                        vocabulary.add(out_str)
                        output.write("%s\n" % out_str)
                        output.flush()

        output.flush()

    print("Done.")


def read_vocab(file_path, antivocab_path=None):
    """
    Reads an output vocabulary.
    :param file_path: Path to the vocabulary file
    :param antivocab_path: Path to the sub-sampled words file (if any)
    :return: (vocabulary as Dict str -> int, reverse_vocabulary as List of str) or (vocabulary, reverse_vocabulary, subsampled_words as List of str) if antivocab_path is specified
    """

    vocab = {"<PAD>": 0, "<UNK>": 1, "<S>": 2, "</S>": 3, "<SUB>": 4}
    reverse_vocab = ["<PAD>", "<UNK>", "<S>", "</S>", "<SUB>"]
    subsampled_words = set()

    with open(file_path, mode="r") as vocab_file:
        for word in vocab_file:
            word = word.strip()
            if word not in vocab:
                vocab[word] = len(vocab)
                reverse_vocab.append(word)

    if antivocab_path is not None:
        with open(antivocab_path) as file:
            for word in file:
                word = word.strip()
                subsampled_words.add(word)

    if antivocab_path is not None:
        return vocab, reverse_vocab, list(subsampled_words)
    else:
        return vocab, reverse_vocab


def merge_vocabs(vocab1, rev_vocab1, vocab2):
    """
    Merges two vocabularies into the first one, keeping the reverse vocabulary consistent.
    :param vocab1: First vocabulary (will contain the merged vocabulary), as Dict str -> int
    :param rev_vocab1: First reverse vocabulary, as List of str
    :param vocab2: Second vocabulary, as Dict str -> int
    :return: (vocab1, rev_vocab1) updated to resemble the merged vocabulary
    """

    v1 = copy.deepcopy(vocab1)
    rv1 = copy.deepcopy(rev_vocab1)

    for key2 in vocab2.keys():
        if key2 not in v1:
            v1[key2] = len(v1)
            rv1.append(key2)

    return v1, rv1


def gold_to_babelnet(gold_file_path, bn_wn_path, output_path):
    """
    Converts a gold.key.txt file in Raganato's format to output BabelNet IDs instead of WordNet sense keys.
    :param gold_file_path: Path to the gold.key.txt file
    :param bn_wn_path: Path to the BabelNet to WordNet mapping
    :param output_path: Path to the file to be written
    :return:
    """

    print("Reading mappings...")
    bn_wn, wn_bn = read_mapping(bn_wn_path)

    print("Parsing started...")
    with \
            open(gold_file_path, mode="r") as gold, \
            open(output_path, mode="w") as output:

        for line in gold:
            line = line.strip().split(" ")
            if len(line) < 2:
                continue

            out_strings = []
            for sense_key in line[1:]:
                wn_id = wn_id_from_sense_key(sense_key)
                out_strings.extend(wn_bn[wn_id])

            output.write("%s %s\n" % (line[0], " ".join(out_strings)))

        output.flush()

    print("Done.")


def gold_to_wnDomains(gold_file_path, bn_wn_path, bn_wnD_path, output_path):
    """
    Converts a gold.key.txt file in Raganato's format to output WordNet Domains instead of WordNet sense keys.
    :param gold_file_path: Path to the gold.key.txt file
    :param bn_wn_path: Path to the BabelNet to WordNet mapping
    :param bn_wnD_path: Path to the BabelNet to WN Domains mapping
    :param output_path: Path to the file to be written
    :return:
    """

    print("Reading mappings...")
    bn_wn, wn_bn = read_mapping(bn_wn_path)
    mapping, rev_mapping = read_mapping(bn_wnD_path)

    print("Parsing started...")
    with \
            open(gold_file_path, mode="r") as gold, \
            open(output_path, mode="w") as output:

        for line in gold:
            line = line.strip().split(" ")
            if len(line) < 2:
                continue

            out_strings = []
            for sense_key in line[1:]:
                wn_id = wn_id_from_sense_key(sense_key)
                for bn_id in wn_bn[wn_id]:
                    out_strings.extend(mapping.get(bn_id, []))

            if len(out_strings) == 0:
                out_strings = ["factotum"]

            output.write("%s %s\n" % (line[0], " ".join(out_strings)))

        output.flush()

    print("Done.")


def gold_to_LEX(gold_file_path, bn_wn_path, bn_LEX_path, output_path):
    """
    Converts a gold.key.txt file in Raganato's format to output lexnames instead of WordNet sense keys.
    :param gold_file_path: Path to the gold.key.txt file
    :param bn_wn_path: Path to the BabelNet to WordNet mapping
    :param bn_LEX_path: Path to the BabelNet to lexnames mapping
    :param output_path: Path to the file to be written
    :return:
    """

    print("Reading mappings...")
    bn_wn, wn_bn = read_mapping(bn_wn_path)
    mapping, rev_mapping = read_mapping(bn_LEX_path)

    print("Parsing started...")
    with \
            open(gold_file_path, mode="r") as gold, \
            open(output_path, mode="w") as output:

        for line in gold:
            line = line.strip().split(" ")
            if len(line) < 2:
                continue

            out_strings = []
            for sense_key in line[1:]:
                wn_id = wn_id_from_sense_key(sense_key)
                for bn_id in wn_bn[wn_id]:
                    out_strings.extend(mapping.get(bn_id, []))

            if len(out_strings) == 0:
                out_strings = ["misc"]

            output.write("%s %s\n" % (line[0], " ".join(out_strings)))

        output.flush()

    print("Done.")


if __name__ == "__main__":
    #make_input_vocab(
    #    train_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #    vocab_path="../resources/semcor.input.vocab.txt",
    #    antivocab_path="../resources/semcor.input.anti.txt"
    #)

    #make_POS_vocab(
    #    train_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
    #    vocab_path="../resources/semcor.POS.txt"
    #)

    #make_output_vocab(
    #    gold_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #    output_path="../resources/semcor.fine_senses.txt"
    #)

    #make_output_vocab(
    #    gold_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #    output_path="../resources/semcor.lexnames.txt",
    #    to_LEX=True,
    #    bn_LEX_path="../resources/babelnet2lexnames.tsv",
    #    bn_wn_path="../resources/babelnet2wordnet.tsv"
    #)

    #make_output_vocab(
    #    gold_file_path="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
    #    output_path="../resources/semcor.wndomains.txt",
    #    to_wnDomains=True,
    #    bn_wnD_path="../resources/babelnet2wndomains.tsv",
    #    bn_wn_path="../resources/babelnet2wordnet.tsv"
    #)

    #v1, rv1 = read_vocab("../resources/semcor.fine_senses.txt")
    #v2, rv2 = read_vocab("../resources/semcor.input.vocab.txt")
    #l1 = len(v1)
    #l2 = len(v2)
    #print("v1: %d" % l1)
    #print("v2: %d" % l2)
    #print("rv2[0]: %s" % rv2[5])
    #print("l1+l2 = %d" % (l1+l2))
    #print("-- merge --")
    #v1, rv1 = merge_vocabs(v1, rv1, v2)
    #print("rv1[l1]: %s" % rv1[l1])
    #print("%d" % len(rv1))

    # --- Evaluation datasets to BabelNet ID, WordNet Domains, and lexnames ---

    #gold_to_babelnet(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                 bn_wn_path="../resources/babelnet2wordnet.tsv",
    #                 output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.bn.txt")

    #gold_to_wnDomains(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #                  bn_wn_path="../resources/babelnet2wordnet.tsv",
    #                  bn_wnD_path="../resources/babelnet2wndomains.tsv",
    #                  output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.wnD.txt")

    #gold_to_LEX(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
    #            bn_wn_path="../resources/babelnet2wordnet.tsv",
    #            bn_LEX_path="../resources/babelnet2lexnames.tsv",
    #            output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.lex.txt")

    #for dataset in ["semeval2013", "semeval2015", "senseval2", "senseval3"]:
    #    print("Converting dataset %s..." % dataset)

    #    print("\tto BabelNet IDs...")
    #    gold_to_babelnet(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.key.txt" % (dataset, dataset),
    #                     bn_wn_path="../resources/babelnet2wordnet.tsv",
    #                     output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.bn.txt" % (dataset, dataset))

    #    print("\tto WordNet Domains...")
    #    gold_to_wnDomains(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.key.txt" % (dataset, dataset),
    #                      bn_wn_path="../resources/babelnet2wordnet.tsv",
    #                      bn_wnD_path="../resources/babelnet2wndomains.tsv",
    #                      output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.wnD.txt" % (dataset, dataset))

    #    print("\tto Lexnames...")
    #    gold_to_LEX(gold_file_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.key.txt" % (dataset, dataset),
    #                bn_wn_path="../resources/babelnet2wordnet.tsv",
    #                bn_LEX_path="../resources/babelnet2lexnames.tsv",
    #                output_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/%s/%s.gold.lex.txt" % (dataset, dataset))

    pass
