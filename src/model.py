import re, unicodedata
import pandas as pd
import numpy as np
import pickle
from joblib import load

class Model():
    def __init__(self, model_path, params_path):
        self.model = load(model_path)
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
        
    def clean(self, s):
        s = s.lower()
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore') \
                    .decode('utf-8', 'ignore')
        s = re.sub('[^a-z]+', ' ', s)
        return s.strip()

    def preprocess(self, data, filter_cols, binary_cols, cat_cols, lencoders, vectorizer):
        df = pd.DataFrame(data)
        X = df[filter_cols].copy()
        for c in binary_cols:
            X[c] = X[c].notnull()
        X.fillna(0, inplace=True)
        for c in cat_cols:
            X[c] = lencoders[c].transform(X[c])
        X['nwords_comment'] = df.comentario.apply(lambda x: len(x.split()))
        X['nupper_comment'] = df.comentario.apply(lambda x: \
                                len([c for c in x if c.isupper()]) / len(x.split()))
        clean_comments = df.comentario.apply(self.clean)
        X_tfidf = vectorizer.transform(clean_comments)
        X_all = np.concatenate([X, X_tfidf.toarray()], axis=1)
        return X_all

    def predict(self, data):
        if data:
            X = self.preprocess(data, **self.params)
            return {
                'predictions': [int(c) for c in self.model.predict(X)]
            }
        return {}