import itertools
import logging

import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from sklearn.externals import joblib

from src.distances import get_most_similar_documents

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


PATH_DICTIONARY = "models/id2word.dictionary"
PATH_CORPUS = "models/corpus.mm"
PATH_LDA_MODEL = "models/LDA.model"
PATH_DOC_TOPIC_DIST = "model/doc_topic_dist.dat"


def head(stream, n=10):
    """
    Return the first `n` elements of the stream, as plain list.
    """
    return list(itertools.islice(stream, n))


def tokenize(text, STOPWORDS):
    # deacc=True to remove punctuations
    return [token for token in simple_preprocess(text, deacc=True)
            if token not in STOPWORDS]


def make_texts_corpus(sentences):
    for sentence in sentences:
        yield simple_preprocess(sentence, deacc=True)


class StreamCorpus(object):
    def __init__(self, sentences, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` documents
        Yield each document in turn, as a list of tokens.
        """
        self.sentences = sentences
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        for tokens in itertools.islice(make_texts_corpus(self.sentences),
                                       self.clip_docs):
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs


class LDAModel:

    def __init__(self, num_topics, passes, chunksize,
                 random_state=100, update_every=1, alpha='auto',
                 per_word_topics=False):
        """
        :param sentences: list or iterable (recommend)
        """

        # data
        self.sentences = None

        # params
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

        # hyperparams
        self.num_topics = num_topics
        self.passes = passes
        self.chunksize = chunksize
        self.random_state = random_state
        self.update_every = update_every
        self.alpha = alpha
        self.per_word_topics = per_word_topics

        # init model
        # self._make_dictionary()
        # self._make_corpus_bow()

    def _make_corpus_bow(self, sentences):
        self.corpus = StreamCorpus(sentences, self.id2word)
        # save corpus
        gensim.corpora.MmCorpus.serialize(PATH_CORPUS, self.corpus)

    def _make_corpus_tfidf(self):
        pass

    def _make_dictionary(self, sentences):
        self.texts_corpus = make_texts_corpus(sentences)
        self.id2word = gensim.corpora.Dictionary(self.texts_corpus)
        self.id2word.filter_extremes(no_below=10, no_above=0.25)
        self.id2word.compactify()
        self.id2word.save(PATH_DICTIONARY)

    def documents_topic_distribution(self):
        doc_topic_dist = np.array(
            [[tup[1] for tup in lst] for lst in self.lda_model[self.corpus]]
        )
        # save documents-topics matrix
        joblib.dump(doc_topic_dist, PATH_DOC_TOPIC_DIST)
        return doc_topic_dist

    def fit(self, sentences):
        self._make_dictionary(sentences)
        self._make_corpus_bow(sentences)
        self.lda_model = gensim.models.LdaModel(
            self.corpus, id2word=self.id2word, num_topics=64, passes=5,
            chunksize=100, random_state=42, alpha=1e-2, eta=0.5e-2,
            minimum_probability=0.0, per_word_topics=False
        )
        self.lda_model.save(PATH_LDA_MODEL)

    def transform(self, sentence):
        """
        :param document: preprocessed document
        """
        document_corpus = next(make_texts_corpus([sentence]))
        corpus = self.id2word.doc2bow(document_corpus)
        document_dist = np.array(
            [tup[1] for tup in self.lda_model.get_document_topics(bow=corpus)]
        )
        return corpus, document_dist

    def predict(self, document_dist):
        doc_topic_dist = self.documents_topic_distribution()
        return get_most_similar_documents(document_dist, doc_topic_dist)

    def update(self, new_corpus):  # TODO
        """
        Online Learning LDA
        https://radimrehurek.com/gensim/models/ldamodel.html#usage-examples
        https://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation
        """
        self.lda_model.update(new_corpus)
        # get topic probability distribution for documents
        for corpus in new_corpus:
            yield self.lda_model[corpus]

    def model_perplexity(self):
        logging.INFO(self.lda_model.log_perplexity(self.corpus))

    def coherence_score(self):
        self.coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(
            model=self.lda_model, texts=self.corpus,
            dictionary=self.id2word, coherence='c_v'
        )
        logging.INFO(self.coherence_model_lda.get_coherence())

    def compute_coherence_values(self, mallet_path, dictionary, corpus,
                                 texts, end=40, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        end : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model
                           with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, end, step):
            model = gensim.models.wrappers.LdaMallet(
                mallet_path, corpus=self.corpus,
                num_topics=self.num_topics, id2word=self.id2word)
            model_list.append(model)
            coherencemodel = gensim.models.coherencemodel.CoherenceModel(
                model=model, texts=self.texts_corpus,
                dictionary=self.dictionary, coherence='c_v'
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def plot(self, coherence_values, end=40, start=2, step=3):
        x = range(start, end, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def print_topics(self):
        pass


def main():
    # TODO
    sentences = None
    sentences = make_texts_corpus(sentences)
    id2word = gensim.corpora.Dictionary(sentences)
    id2word.filter_extremes(no_below=20, no_above=0.1)
    id2word.compactify()

    # save dictionary
    # id2word.save('path_to_save_file.dictionary')
    cospus = StreamCorpus(sentences, id2word)
    # save corpus
    # gensim.corpora.MmCorpus.serialize('path_to_save_file.mm', cospus)
    # load corpus
    # mm_corpus = gensim.corpora.MmCorpus('path_to_save_file.mm')
    lda_model = gensim.models.LdaModel(
        cospus, num_topics=32, id2word=id2word, passes=10, chunksize=100
    )
    # save model
    # lda_model.save('path_to_save_model.model')
    lda_model.print_topics(-1)


if __name__ == '__main__':
    main()
