from lxml import etree
from collections import namedtuple

import utils as u


XMLEntry = namedtuple("XMLEntry", "id lemma pos has_instance")
GoldEntry = namedtuple("GoldEntry", "id senses")


class TrainParser:
    """
    Class to parse XML training files compliant to Raganato's format.
    """

    def __init__(self, file):
        self.file = file

    def __enter__(self):
        self.xml = open(self.file, mode="rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.xml.close()

    def next_sentence(self):
        """
        Parses sentences from the XML corpus.
        :return: Sentence broken down to id, lemma, and pos for each word, as List of Entry in a generator fashion
        """
        for event, sentence in etree.iterparse(self.xml, tag="sentence"):
            sent = []

            if event == "end":
                for element in sentence:
                    if element.tag == "wf":
                        lemma = element.get("lemma")
                        pos = element.get("pos")
                        entry = XMLEntry(id=None, lemma=lemma, pos=pos, has_instance=False)
                        sent.append(entry)

                    elif element.tag == "instance":
                        iid = element.get("id")
                        lemma = element.get("lemma")
                        pos = element.get("pos")
                        entry = XMLEntry(id=iid, lemma=lemma, pos=pos, has_instance=True)
                        sent.append(entry)

                        # update XMLEntry in the first position with has_instance=True for efficiency purpose
                        if not sent[0].has_instance:
                            first_entry = sent[0]
                            sent[0] = XMLEntry(id=first_entry.id,
                                               lemma=first_entry.lemma,
                                               pos=first_entry.pos,
                                               has_instance=True)

                yield sent

                sentence.clear()


class GoldParser:
    """
    Class to parse gold.key.txt files compliant to Raganato's format.
    """

    def __init__(self, file):
        self.file = file
        self.forgotten_sent_id = None
        self.forgotten_entry = None

    def __enter__(self):
        self.handle = open(self.file, mode="r") if self.file != "" else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.close()

    def next_sentence(self):
        """
        Parses sentences from the gold.key.txt corpus.
        :return: Sentence broken down to id and senses for each word, as List of Entry in a generator fashion
        """

        sent = []
        prev_sent_id = None
        for line in self.handle:
            line = line.strip()
            line = line.split(" ")
            if len(line) < 2:
                continue

            iid = line[0]
            sent_id = ".".join(iid.split(".")[:-1])

            senses = set()
            for sense in line[1:]:
                senses.add(u.wn_id_from_sense_key(sense))

            entry = GoldEntry(id=iid, senses=list(senses))

            if self.forgotten_entry is not None and len(sent) == 0:
                sent.append(self.forgotten_entry)
                if self.forgotten_sent_id != sent_id:
                    prev_sent_id = self.forgotten_sent_id
                    self.forgotten_sent_id = sent_id
                    self.forgotten_entry = entry

                    yield sent

            if prev_sent_id is None or prev_sent_id == sent_id:
                sent.append(entry)
                prev_sent_id = sent_id
            else:
                self.forgotten_sent_id = sent_id
                self.forgotten_entry = entry
                yield sent


if __name__ == "__main__":
    with TrainParser("../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml") as p:
        max = 4
        for s in p.next_sentence():
            if max == 0:
                break

            print(s)
            max -= 1

    print("---")
    with GoldParser("../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt") as p:
        min = 46
        max = 6
        for i in range(min+max):
            s = next(p.next_sentence())
            if i > min:
                print(s)

    pass
