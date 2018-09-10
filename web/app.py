import os
import logging
import random

from flask import Flask, jsonify, render_template
import numpy as np
import pymongo

from distances import get_most_similar_documents
from models import make_texts_corpus
from utils import markdown_to_text

import settings

client = pymongo.MongoClient(settings.MONGODB_SETTINGS["host"])
db = client[settings.MONGODB_SETTINGS["db"]]
mongo_col = db[settings.MONGODB_SETTINGS["collection"]]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "framgia123")

# app.config.from_object('web.config.DevelopmentConfig')
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def load_model():
    import gensim  # noqa
    from sklearn.externals import joblib  # noqa
    # load LDA model
    lda_model = gensim.models.LdaModel.load(
        settings.PATH_LDA_MODEL
    )
    # load corpus
    corpus = gensim.corpora.MmCorpus(
        settings.PATH_CORPUS
    )
    # load dictionary
    id2word = gensim.corpora.Dictionary.load(
        settings.PATH_DICTIONARY
    )
    # load documents topic distribution matrix
    doc_topic_dist = joblib.load(
        settings.PATH_DOC_TOPIC_DIST
    )

    return lda_model, corpus, id2word, doc_topic_dist


lda_model, corpus, id2word, doc_topic_dist = load_model()


@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify({
        'call': 'success',
        'message': 'pong!'
    })


@app.route('/posts/', methods=["GET"])
def show_posts():
    idrss = random.sample(range(0, 14157), 10)
    posts = mongo_col.find({"idrs": {"$in": idrss}})
    random_posts = [
        {
            "url": post["canonical_url"],
            "title": post["title"],
            "slug": post["slug"]
        }
        for post in posts
    ]
    return render_template('index.html', random_posts=random_posts)


@app.route('/posts/<slug>', methods=["GET"])
def show_post(slug):
    main_post = mongo_col.find_one({"slug": slug})
    main_post = {
        "url": main_post["canonical_url"],
        "title": main_post["title"],
        "slug": main_post["slug"],
        "content": main_post["contents"]
    }

    # preprocessing
    content = markdown_to_text(main_post["content"])
    text_corpus = make_texts_corpus([content])
    bow = id2word.doc2bow(next(text_corpus))
    doc_distribution = np.array(
        [doc_top[1] for doc_top in lda_model.get_document_topics(bow=bow)]
    )

    # recommender posts
    most_sim_ids = list(get_most_similar_documents(
        doc_distribution, doc_topic_dist))[1:]

    logging.INFO(most_sim_ids)
    most_sim_ids = [int(id_) for id_ in most_sim_ids]
    posts = mongo_col.find({"idrs": {"$in": most_sim_ids}})
    related_posts = [
        {
            "url": post["canonical_url"],
            "title": post["title"],
            "slug": post["slug"]
        }
        for post in posts
    ][1:]

    return render_template(
        'index.html', main_post=main_post, posts=related_posts
    )


@app.route('/posts_HAU/<slug>', methods=["GET"])
def show_post_HAU(slug):
    """
    Author: Thanh Hau
    """
    from sklearn.externals import joblib  # noqa
    sim_topics = joblib.load('data/similarity_dict_HAU.pkl')
    main_post = mongo_col.find_one({"slug": slug})
    main_post = [
        {
            "url": main_post["canonical_url"],
            "title": main_post["title"],
            "slug": main_post["slug"],
            "content": main_post["contents"]
        }
    ]
    main_post = main_post[0]

    most_sim_slugs = sim_topics[slug]
    posts = mongo_col.find({"slug": {"$in": most_sim_slugs}})
    related_posts = [
        {
            "url": post["canonical_url"],
            "title": post["title"],
            "slug": post["slug"]
        }
        for post in posts
    ]

    return render_template(
        'index.html', main_post=main_post, posts=related_posts
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
