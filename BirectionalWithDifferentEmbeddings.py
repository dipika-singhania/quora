import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import load_model
import argparse
import sys


class QuoraData:
    def __init__(self, args):
        self.args = args

    def readFileAndFormDataFrames(self):
        train_df = pd.read_csv(self.args.input_dir + self.args.train_input_file)
        self.test_df  = pd.read_csv(self.args.input_dir + self.args.test_input_file)
        self.fObj.write("Train shape : {0} \n".format(train_df.shape))
        self.fObj.write("Test shape : {0} \n".format(self.test_df.shape))

        ## split to train and val
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

        ## fill up the missing values
        self.train_X = train_df["question_text"].values
        self.val_X = val_df["question_text"].values
        self.test_X = self.test_df["question_text"].values

        ## Tokenize the sentences
        self.tokenizer = Tokenizer(num_words=self.args.max_features)
        self.tokenizer.fit_on_texts(list(self.train_X))
        self.train_X = self.tokenizer.texts_to_sequences(self.train_X)
        self.val_X = self.tokenizer.texts_to_sequences(self.val_X)
        self.test_X = self.tokenizer.texts_to_sequences(self.test_X)

        ## Pad the sentences
        self.train_X = pad_sequences(self.train_X, maxlen=self.args.max_len)
        self.val_X = pad_sequences(self.val_X, maxlen=self.args.max_len)
        self.test_X = pad_sequences(self.test_X, maxlen=self.args.max_len)

        ## Get the target values
        self.train_y = train_df['target'].values
        self.val_y = val_df['target'].values

    def applyFloat(self, a):
        try:
            return np.asarray(a, dtype='float32')
        except:
            return np.nan

    def applyMean(self, a):
        try:
            return np.mean(a)
        except:
            return np.nan

    def applyStd(self, a):
        try:
            return np.std(a)
        except:
            return np.nan

    def createEmbedMatrix(self, Embedding_File, nb_words):
        read_embed = pd.read_csv(Embedding_File, sep="\s+", encoding="utf-8", header=None, error_bad_lines=False)
        read_embed = read_embed.set_index(0).T
        read_embed.apply(lambda a: np.asarray(a, dtype='float32'))
        emb_mean = read_embed.apply(np.mean, axis=1)
        emb_std = read_embed.apply(np.std, axis=1)
        embed_size = read_embed.shape[0]
        word_index = self.tokenizer.word_index
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        self.fObj.write("embedding matrix size = {0}".format(embedding_matrix.shape))
        for word, i in word_index.items():
            if i >= self.args.max_features:
                continue
            try:
                embedding_vector = read_embed[word]
            except:
                continue
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def createFitSaveModel(self, embedding_size, embedding_matrix, model_file):
        inp = Input(shape=(self.args.max_len,))
        x = None
        if embedding_matrix is not None:
            x = Embedding(embedding_size, self.args.embed_size, weights=[embedding_matrix])(inp)
        else:
            x = Embedding(embedding_size, self.args.embed_size)(inp)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(self.train_X, self.train_y, batch_size=512, epochs=self.args.epoch, validation_data=(self.val_X, self.val_y))
        model.save(model_file)  # creates a HDF5 file 'my_model.h5'
        return model

    def loadSavedModel(self, model_file):
        try:
            model = load_model(model_file)
        except:
            print("Unable to find model file")
            sys.exit(1)
        return model

    def predictAndPrintValScore(self, datafrm, model):
        return model.predict([datafrm], batch_size=1024, verbose=1)

    def calculateAndPrintF1Score(self, actual, predicted):
        maxF1 = 0
        max_threshold = 0
        for threshold in np.arange(0.25, 0.60, 0.01):
            threshold = np.round(threshold, 2)
            f1Score = metrics.f1_score(actual, (predicted > threshold).astype(int))
            if (f1Score > maxF1):
                maxF1 = f1Score
                max_threshold = threshold
            self.fObj.write("F1 score validation data for threshold {0} is {1}\n".format(threshold, f1Score))
        return max_threshold

    def printSubmissionFile(self, predicted_values, calculated_threshold):
        pred_test_y = (predicted_values > calculated_threshold).astype(int)
        out_df = pd.DataFrame({"qid": self.test_df["qid"].values})
        out_df['prediction'] = pred_test_y
        out_df.to_csv(self.output_submission_file, index=False)

    def createFilesName(self, args, embed):
        self.output_log_name = args.output_dir + "output_log_"
        self.model_file = args.output_dir + "model_"
        self.output_submission_file = args.output_dir + "submission_"
        sub_script = embed
        sub_script = sub_script + "_" + str(args.max_features) + "_"
        sub_script = sub_script + "_" + str(args.embed_size) + "_"
        sub_script = sub_script + "_" + str(args.max_len) + "_"
        self.output_log_name = self.output_log_name + sub_script + ".log"
        self.model_file = self.model_file + sub_script + ".h5"
        self.output_submission_file = self.output_submission_file + sub_script + ".csv"
        self.fObj = open(self.output_log_name, "w")

    def runSingleEmbedding(self):
        embeddingFile = self.args.embed_dir
        if (self.args.useGlove is True):
            embeddingFile = embeddingFile + self.args.glove_embed_file
            self.createFilesName(self.args, "_glove_")
            print("Use glove embedding")
        elif (self.args.useWiki is True):
            embeddingFile = embeddingFile + self.args.wiki_embed_file
            self.createFilesName(self.args, "_wiki_")
            print("Use wiki embedding")
        elif (self.args.useGoogle is True):
            embeddingFile = embeddingFile + self.args.google_embed_file
            self.createFilesName(self.args, "_google_")
            print("Use google embedding")
        elif (self.args.useParagram is True):
            embeddingFile = embeddingFile + self.args.paragram_embed_file
            self.createFilesName(self.args, "_paragram_")
            print("Use paragram embedding")
        else:
            embeddingFile = None
            self.createFilesName(self.args, "_normal_")

        self.readFileAndFormDataFrames()
        model = None
        if self.args.do_train is True:
            embedding_matrix = None
            nb_words = self.args.max_features
            if embeddingFile is not None:
                nb_words = min(self.args.max_features, len(self.tokenizer.word_index))
                embedding_matrix = self.createEmbedMatrix(embeddingFile, nb_words)
            model = self.createFitSaveModel(nb_words, embedding_matrix, self.model_file)
        else:
            model = self.loadSavedModel(self.model_file)
        normal_pred_output = self.predictAndPrintValScore(self.val_X, model)
        calculated_thresold = self.calculateAndPrintF1Score(self.val_y, normal_pred_output)
        self.printSubmissionFile(self.predictAndPrintValScore(self.test_X, model), calculated_thresold)
        pass

    def runMixedEmdedding(self):
        embeddingFile = self.args.embed_dir + self.args.glove_embed_file
        self.createFilesName(self.args, "_glove_")
        embedding_matrix = self.createEmbedMatrix(embeddingFile)
        self.readFileAndFormDataFrames()
        model = None
        if self.args.do_train is True:
            model = self.createFitSaveModel(self.args.max_features, embedding_matrix, self.model_file)
        else:
            model = self.loadSavedModel(self.model_file)
        val_pred_output_glove = self.predictAndPrintValScore(self.val_X, model)
        pred_test_glove = self.predictAndPrintValScore(self.text_X)

        embeddingFile = self.args.embed_dir + self.args.wiki_embed_file
        self.createFilesName(self.args, "_wiki_")
        embedding_matrix = self.createEmbedMatrix(embeddingFile)
        model = None
        if self.args.do_train is True:
            model = self.createFitSaveModel(self.args.max_features, embedding_matrix, self.model_file)
        else:
            model = self.loadSavedModel(self.model_file)
        val_pred_output_wiki = self.predictAndPrintValScore(self.val_X, model)
        pred_test_wiki = self.predictAndPrintValScore(self.text_X)

        embeddingFile = self.args.embed_dir + self.args.paragram_embed_file
        self.createFilesName(self.args, "_paragram_")
        embedding_matrix = self.createEmbedMatrix(embeddingFile)
        model = None
        if self.args.do_train is True:
            model = self.createFitSaveModel(self.args.max_features, embedding_matrix, self.model_file)
        else:
            model = self.loadSavedModel(self.model_file)
        val_pred_output_paragram = self.predictAndPrintValScore(self.val_X, model)
        pred_test_paragram = self.predictAndPrintValScore(self.text_X)

        val_pred_output = (0.33 * val_pred_output_glove) + (0.33 * val_pred_output_wiki) + (0.34 * val_pred_output_paragram)
        test_pred_output = (0.33 * pred_test_glove) + (0.33 * pred_test_wiki) + (0.34 * pred_test_paragram)
        calculated_thresold = self.calculateAndPrintF1Score(self.val_y, val_pred_output)
        self.printSubmissionFile(test_pred_output, calculated_thresold)

def main():
    argObj = argparse.ArgumentParser()
    argObj.add_argument("--input_dir", default="data/")
    argObj.add_argument("--output_dir", default="output/")
    argObj.add_argument("--embed_dir", default="embeddings/")
    argObj.add_argument("--train_input_file", default="train/train.csv")
    argObj.add_argument("--test_input_file", default="test/test.csv")
    argObj.add_argument("--glove_embed_file", default="glove.840B.300d/glove.840B.300d.txt")
    argObj.add_argument("--wiki_embed_file", default="wiki-news-300d-1M/wiki-news-300d-1M.vec")
    argObj.add_argument("--google_embed_file", default="GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin")
    argObj.add_argument("--paragram_embed_file", default="paragram_300_sl999/paragram_300_sl999.txt")
    argObj.add_argument("--no_embed", default=True, action='store_true')
    argObj.add_argument("--useGlove", default=False, action='store_true')
    argObj.add_argument("--useWiki", default=False, action='store_true')
    argObj.add_argument("--useGoogle", default=False, action='store_true')
    argObj.add_argument("--useParagram", default=False, action='store_true')
    argObj.add_argument("--useMixed", default=False, action='store_true')
    argObj.add_argument("--do_train", default=True)
    argObj.add_argument("--max_features", default=50000, type=int)
    argObj.add_argument("--embed_size", default=300, type=int)
    argObj.add_argument("--max_len", default=60, type=int)
    argObj.add_argument("--epoch", default=4, type=int)
    args = argObj.parse_args()
    quoraObj = QuoraData(args)

    if args.useGlove is True or args.useWiki is True or args.useGoogle is True or args.useParagram is True:
        quoraObj.runSingleEmbedding()
    elif args.useMixed:
        quoraObj.runMixedEmbedding()
    else:
        quoraObj.runSingleEmbedding()


if __name__ == '__main__':
    main()

