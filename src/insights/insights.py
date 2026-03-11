from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(docs):

    vectorizer = TfidfVectorizer(max_features=10)

    X = vectorizer.fit_transform(docs)

    return vectorizer.get_feature_names_out()