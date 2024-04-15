import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from collections import Counter

def feature_engineering(df):

    # get word position percentage
    max_indices = df.groupby(['activityId', 'phraseIndex'])['word_index'].max().reset_index()
    max_indices.rename(columns={'word_index': 'phrase_length'}, inplace=True)
    df = pd.merge(df, max_indices, how='left', on=['activityId', 'phraseIndex'])
    df['phrase_length'] = df['phrase_length'] + 1
    df['word_index'] = (df['word_index']/ df['phrase_length'])

    # correct score rate
    df['overall_correct_score'] = (df['amazon_correct'] + df['kaldi_correct'] + df['kaldina_correct']) / 3

    # correct score rate with confidence
    df['overall_correct_score_confidence'] = (df['amazon_correct'] * df['amazon_confidence'] \
                                            + df['kaldi_correct'] * df['kaldi_confidence'] \
                                            + df['kaldina_correct'] * df['kaldina_confidence']) / 3
    
    # substitute score rate
    df['overall_substituted_score'] = (df['amazon_substituted'] + df['kaldi_substituted'] + df['kaldina_substituted']) / 3

    # substitute score rate
    df['overall_deleted_score'] = (df['amazon_deleted'] + df['kaldi_deleted'] + df['kaldina_deleted']) / 3

    # substitute lapse rate
    df['overall_lapse'] = (df['amazon_lapse'] + df['kaldi_lapse'] + df['kaldina_lapse']) / 3

    # pos
    df['pos_tags'] = df['pos_tags'].apply(lambda x: 1 if x in ['PROPN', 'X', 'NOUN_Plur', 'VERB_Fin'] else 0)

    return df

def feature_selection(X_train, y_train, X_test, k=22):
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected


def run_experiment(X_train_selected, y_train):

    print('Start Training...')

    # Random Forest
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train_selected, y_train)

    # XGBoost
    counter = Counter(y_train)
    estimate = counter[0] / counter[1]
    model_xgb = xgb.XGBClassifier(scale_pos_weight=estimate, use_label_encoder=False, eval_metric='logloss', alpha=0.5)
    model_xgb.fit(X_train_selected, y_train)

    # Deep Neural Network
    model_nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_selected.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_nn.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_nn.fit(X_train_selected, y_train, epochs=20, batch_size=10)

    return model_rf, model_xgb, model_nn

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print('roc_auc:', roc_auc_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


def main(args):

    data_df = pd.read_csv(args.dataset_path)
    data_df = data_df.fillna(0)
    data_df = feature_engineering(data_df)
    data = data_df.select_dtypes(include='number')

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_selected, X_test_selected = feature_selection(X_train, y_train, X_test)

    # experiment
    model_rf, model_xgb, model_nn = run_experiment(X_train_selected, y_train)

    print("Evaluation:\n")

    # Evaluate Random Forest
    print('Random Forest')
    evaluate_model(model_rf, X_test_selected, y_test)
    print('\n')

    # Evaluate XGBoost
    print('XGBoost')
    evaluate_model(model_xgb, X_test_selected, y_test)
    print('\n')

    # Evaluate Neural Network
    print('NN')
    predictions_nn = (model_nn.predict(X_test_selected) > 0.6).astype('int32')
    print(classification_report(y_test, predictions_nn))
    print('roc_auc:', roc_auc_score(y_test, predictions_nn))
    print(confusion_matrix(y_test, predictions_nn))
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="data.csv",help='path to dataset')
    args = parser.parse_args()

    main(args)

