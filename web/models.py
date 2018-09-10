from mongoengine import connect, Document, StringField


connect(db="rsframgia", host="mongodb://localhost:27017")


class Books(Document):
    id_ = StringField()
    slug = StringField()
    title = StringField()
    user_id = StringField()
    canonical_url = StringField()
    contents = StringField()
    idrs = StringField()
