# This is a sample Python script.
import numpy as np
import util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    text_lower = [t.lower() for t in text]
    text_tokens = [word_tokenize(t) for t in text_lower]
    text_tokens_stop = []

    for sent in tqdm(text_tokens):
        text_tokens_stop.append([ps.stem(w) for w in sent if w not in stop_words and w.isalpha()])

    return text_tokens_stop


def function_words_features(tokens):
    function_words = util.load_function_words("ewl_function_words.txt")
    X = np.zeros(shape=(len(tokens), len(function_words)))
    for i, s in tqdm(enumerate(tokens)):
        for t in s:
            for j, fw in enumerate(function_words):
                if fw == t:
                    X[i][j] += 1
    return X


def structure_features(tokens):
    tokens_len = []
    for row in tokens:
        tmp = []
        for token in row:
            tmp.append(len(token))
        tokens_len.append(tmp)

    X_structure = np.zeros(shape=(len(tokens), 7))
    for i in range(len(tokens_len)):
        row = tokens_len[i]
        tmp = np.asarray(row)
        X_structure[i][0] = np.mean(tmp)
        X_structure[i][1] = np.max(tmp)
        X_structure[i][2] = np.min(tmp)
        X_structure[i][3] = np.percentile(tmp, 10)
        X_structure[i][4] = np.percentile(tmp, 20)
        X_structure[i][5] = np.percentile(tmp, 80)
        X_structure[i][6] = np.percentile(tmp, 90)

    return X_structure


def ngram_features(train, test, n=2, features=500):
    vectorizer = CountVectorizer(ngram_range=(n, n), max_features=features)
    train_corpus = [" ".join(token) for token in train]
    test_corpus = [" ".join(token) for token in test]
    vectorizer.fit(train_corpus + test_corpus)
    X_train = vectorizer.transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    return X_train, X_test


def confusion_matrix(model, X_test, y_test, le):
    y_pred = le.inverse_transform(model.predict(X_test))
    y_true = le.inverse_transform(y_test)
    print(metrics.classification_report(y_true, y_pred, digits=3))
    return


def main():
    texts, authors = util.import_data('./C50/C50train')
    test_texts, y_test = util.import_data('./C50/C50test')
    all_labels = authors + y_test
    # print(set(authors) == set(y_test))

    le = preprocessing.LabelEncoder()
    all_labels = le.fit_transform(all_labels)
    all_texts = texts + test_texts

    cleaned_tokens = preprocess(all_texts)
    print(len(cleaned_tokens), len(all_labels))
    cleaned_tokens, labels = util.shuffle(cleaned_tokens, all_labels)

    X_train, X_test, y_train, y_test = train_test_split(cleaned_tokens, labels, test_size=0.3)

    print("Structure Features")
    X_structure = structure_features(X_train)
    X_structure_test = structure_features(X_test)
    clf_svc = svm.SVC(decision_function_shape='ovo')
    clf_svc.fit(X_structure, y_train)
    print("Train:", clf_svc.score(X_structure, y_train))
    print("Test:", clf_svc.score(X_structure_test, y_test))
    clf_log = LogisticRegression(random_state=0).fit(X_structure, y_train)
    print("log train:", accuracy_score(y_train, clf_log.predict(X_structure)))
    print("log test:", accuracy_score(y_test, clf_log.predict(X_structure_test)))

    print("Function Word:")
    X_function = function_words_features(X_train)
    X_test_function = function_words_features(X_test)
    clf_svc = svm.SVC(decision_function_shape='ovo')
    clf_svc.fit(X_function, y_train)
    print("Train:", clf_svc.score(X_function, y_train))
    print("Test:", clf_svc.score(X_test_function, y_test))
    clf_log = LogisticRegression(random_state=0).fit(X_function, y_train)
    print("log train:", accuracy_score(y_train, clf_log.predict(X_function)))
    print("log test:", accuracy_score(y_test, clf_log.predict(X_test_function)))

    print("Frequent tokens")
    X_freq, X_freq_test = ngram_features(X_train, X_test, n=1)
    clf_svc_ngram = svm.SVC(decision_function_shape='ovo')
    clf_svc_ngram.fit(X_freq, y_train)
    print("Train:", clf_svc_ngram.score(X_freq, y_train))
    print("Test:", clf_svc_ngram.score(X_freq_test, y_test))

    print("2-gram")
    X_2gram, X_2gram_test = ngram_features(X_train, X_test)
    clf_svc_ngram = svm.SVC(decision_function_shape='ovo')
    clf_svc_ngram.fit(X_2gram, y_train)
    print("Train:", clf_svc_ngram.score(X_2gram, y_train))
    print("Test:", clf_svc_ngram.score(X_2gram_test, y_test))

    print("3-gram")
    X_3gram, X_3gram_test = ngram_features(X_train, X_test, 3)
    clf_svc_ngram = svm.SVC(decision_function_shape='ovo')
    clf_svc_ngram.fit(X_3gram, y_train)
    print("Train:", clf_svc_ngram.score(X_3gram, y_train))
    print("Test:", clf_svc_ngram.score(X_3gram_test, y_test))

    return


if __name__ == '__main__':
    main()






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
