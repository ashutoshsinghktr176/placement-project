import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

def build_dataset():
    data = [
        (["python","machine learning"], "A"),
        (["aws","docker"], "A"),
        (["react","javascript"], "B"),
        (["java","sql"], "C"),
    ]
    return data

def train_model():
    data = build_dataset()
    X_raw = [d[0] for d in data]
    y_raw = [d[1] for d in data]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_raw)

    mapping = {"A":2,"B":1,"C":0}
    y = [mapping[v] for v in y_raw]

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    def predict(skills):
        X_new = mlb.transform([skills])
        pred = clf.predict(X_new)[0]
        prob = max(clf.predict_proba(X_new)[0])
        inv = {2:"A",1:"B",0:"C"}
        return inv[pred], float(prob)

    return {"clf":clf,"mlb":mlb,"predict":predict}