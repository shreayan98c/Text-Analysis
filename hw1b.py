import re
import argparse
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class SegmentClassifier:
    def train(self, trainX, trainY):
        self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        # numbers = sum(c.isalpha() for c in text) / len(text)
        # letters = sum(c.isalpha() for c in text) / len(text)
        # spaces = sum(c.isspace() for c in text) / len(text)
        # others = (len(text) - numbers - letters - spaces) / len(text)

        words = text.split()

        # check if the text is a mail
        try_find_address = False
        if re.match(
            '^From:|^Article:|^Path:|^Newsgroups:|^Subject:|^Date:|^Organization:|^Lines:|^Approved:|^Message-ID:|^References:',
            words[0]):
            try_find_address = True

        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),
            # numbers,
            # letters,
            # spaces,
            # others,
            sum(1 if re.match('^(>|:|\s*\S*\s*>|@)', word)  # quotation starts
                     or re.match('^.+(wrote|writes|said):', word) else 0 for word in words),  # article start
            text.count(' '),  # number of spaces to detect blank lines
            text.count('|') + text.count('-') + text.count('+') + text.count('_') + text.count('\\') + text.count('/'),  # figures
            sum(1 if w.isupper() else 0 for w in words) / len(words),  # uppercase chars ratio
            sum(1 if w.isnumeric() else 0 for w in words) / len(words),  # numeric chars ratio
            1 if try_find_address else 0,  # if any "headline" keyword found
            # len(re.sub('[\w]+', '', text)),  # number of non word chars (special chars)
            len(re.sub('[\w]+', '', text)),  # number of non word chars (special chars)
            sum(1 if re.match('^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$', word)  # for emails
                     or re.match('^\D?(\d{3})\D?\D?(\d{3})\D?(\d{4})$', word) else 0 for word in words),  # for phone numberss
                                                                               #
            # text.count(''),

        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()