# BNIRQA
The BNIRQA model : A Bayesian Network based Information Retrieval Model for Question Answering.

This project suggests an effective method to deal with open-domain question answering mainly by
relying on Wikipedia, as a prime source of knowledge. The first step in this approach was to
retrieve a collection of most relevant articles to a factoid question of the Stanford Question answering
dataset (SQuAD) [1], based on a preliminary search on a predefined dataset (Wikipedia),
and then dynamically create a probabilistic information retrieval model (BNIRM) in order to find
the most relevant documents to the query, based on the words dependency in a predefined dataset
(Wikipedia articles) using Bayesian networks (BNs).
