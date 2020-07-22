from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from csv import reader, writer
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, train_file, test_file, result_file_location):
        self.train_file = train_file
        self.test_file = test_file
        self.result_file_location = result_file_location

    def classify(self):
        X, Y = self.__get_training_data()

        X = self.__extract_hour_feature(X, [5, 6])

        X = self.__remove_unwanted_cols(X, [7, 8, 9, 10])
        X_test, ids = self.__get_test_data()
        X_test = self.__extract_hour_feature(X_test, [5, 6])
        X_test = self.__remove_unwanted_cols(X_test, [7, 8, 9, 10])
        Y_binary = []
        for flag in Y:
            if flag == 'correct':
                Y_binary.append(1)
            else:
                Y_binary.append(0)
        Y = Y_binary

        # For decision tree
        result = self.train_and_predict_dt(X, Y, X_test)
        result = zip(ids, result)
        self.__write_results(self.result_file_location + str(datetime.datetime.now()) + 'dt_results.csv', result)

        # For random forest
        result = self.train_and_predict_rf(X, Y, X_test)
        result = zip(ids, result)
        self.__write_results(self.result_file_location + str(datetime.datetime.now()) + 'rf_results.csv', result)

        # For NB
        result = self.train_and_predict_nb(X, Y, X_test)
        result = zip(ids, result)
        self.__write_results(self.result_file_location + str(datetime.datetime.now()) + 'nb_results.csv', result)

    def train_and_predict_dt(self, X, Y, X_test):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)

        result = clf.predict(X_test)
        return result

    def train_and_predict_rf(self, X_train, Y_train, X_test):
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
        clf.fit(X_train, Y_train)
        result = clf.predict(X_test)
        return result

    def train_and_predict_nb(self, X_train, Y_train, X_test):
        clf = GaussianNB()
        clf.fit(X_train, Y_train)
        result = clf.predict(X_test)
        return result

    def __extract_hour_feature(self, X, col_indices):
        new_X = []
        for row in X:
            for col_index in col_indices:
                row[col_index] = datetime.datetime.strptime(row[col_index], "%m/%d/%Y %H:%M").hour
            new_X.append(row)
        return new_X

    def __read_data_file(self, file):
        # read csv file as a list of lists
        with open(file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_of_rows = list(csv_reader)
        return list_of_rows

    def __write_results(self, file, result):
        with open(file, 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = writer(myfile)
            wr.writerow(("tripid", "prediction"))
            wr.writerows(result)

    def __get_training_data(self):
        X = self.__read_data_file(TRAIN_FILE)
        X = X[1:]
        X_tmp = []
        Y = []
        for line in X:
            # Remove None
            if "" in line:
                continue
            X_tmp.append(line[1:-1])
            Y.append(line[-1])
        X = X_tmp
        return X, Y

    def __remove_unwanted_cols(self, X, col_indices):
        new_X = []
        for row in X:
            new_row = []
            for col_index in range(0, len(row)):
                if col_index not in col_indices:
                    new_row.append(float(row[col_index]))
            new_X.append(new_row)
        return new_X

    def __get_test_data(self):
        X_test = self.__read_data_file(TEST_FILE)
        X_test = X_test[1:]
        X_test_temp = []
        ids = []
        for line in X_test:
            X_test_temp.append(line[1:])
            ids.append(line[0])
        X_test = X_test_temp
        return X_test, ids


class Analyzer():

    def __init__(self, file):
        self.data = pd.read_csv(file)

    def analyze(self):
        # Summary stats of attributes
        print('Attribute types: \n' + str(self.data.dtypes) + '\n')
        print('Description of data: \n' + str(self.data.describe()))

        # Missing values
        print(self.data.isna().any())

        # invalid, out of range values
        hist = self.data.hist(bins=10)
        plt.subplots_adjust(hspace=0.8)
        plt.show()


TRAIN_FILE = '../data_files/train.csv'
TEST_FILE = '../data_files/test.csv'
RESULT_FILE_LOCATION = '../data_files/'

if __name__ == "__main__":
    analyzer = Analyzer(TRAIN_FILE)
    analyzer.analyze()

    classifier = Classifier(TRAIN_FILE, TEST_FILE, RESULT_FILE_LOCATION)
    classifier.classify()
