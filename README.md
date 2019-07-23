# NLP 2018/2019 ([more][1])

## Homework 3

Natural language comprises a huge amount of words which have multiple meanings and are therefore ambiguous. The human brain is very capable at solving such ambiguities, while for a machine it may be a very though challenge.

In the NLP community, word sense disambiguation (WSD) is the task of automatically selecting the most appropriate sense for a given word in a given context, be it a sentence or a whole document, among all the possible senses which can be associated to that word for the particular part-of-speech (POS) tag considered.

Traditionally, WSD has been tackled via many approaches, some of them being: always using the most frequent sense (MFS); sense with the greatest overlap of definitions w.r.t. a context window (Lesk); sense with the highest score given by a random graph walk (or personalized PageRank) when exploiting a knowledge graph (UKB).

Recently, with the advent of machine learning, and more specifically with deep learning, neural network approaches have become increasingly popular: typically, neural WSD is performed as single classification tasks for each word, considering a sliding window over the context; although quite efficient, this approach doesn't explicitly consider the sequential nature of sense distributions over the context.

An example of directly addressing such sequential nature of senses can be the one presented in **Neural Sequence Learning Models for Word Sense Disambiguation**, which is the reference paper of this homework assignment. In the reference paper, the authors suggest to tackle WSD as means of sequence tagging, possibly exploiting multi-task learning by also making the model perform related tasks like POS tagging.

This homework assignment has the target of performing WSD at different levels: other than over the fine-grained sense inventory of WordNet (which has been mapped to BabelSynsets), it is also required to implement coarse-grained WSD over WordNet Domains and Lexicographer's names (Lexnames), both of them being mapped to BabelSynsets as well.

[Continue reading][2]

[1]: http://naviglinlp.blogspot.com/
[2]: ./hw3_report_anonymous.pdf
