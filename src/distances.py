import numpy as np
from scipy.stats import entropy


def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M (the number of documents in the corpus)
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix

    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))


def get_most_similar_documents(query, matrix, k=10):
    """
    This function implements the Jensen-Shannon distance above
    and returns the top k indices of the smallest jensen shannon distances
    """
    # list of jensen shannon distances
    sims = jensen_shannon(query, matrix)
    # the top k positional index of the smallest Jensen Shannon distances
    return sims.argsort()[:k]
