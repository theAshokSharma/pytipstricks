import itertools
import re
from typing import Iterable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# To compute the accuracy of models
from surprise import accuracy

# class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader

# class for loading datasets
from surprise.dataset import Dataset

# for tuning model hyperparameters
from surprise.model_selection import GridSearchCV

# for splitting the rating data in train and test dataset
from surprise.model_selection import train_test_split

# for implementing similarity-based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# for implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD

# for implementing KFold cross-validation
from surprise.model_selection import KFold

#For implementing clustering-based recommendation system
from surprise import CoClustering



def missing_values(df):
    """
    returns missing data in data frame as total and percentage
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def unique_values_count(df: pd.DataFrame, cat_col=None):
    """
    print unique_values count of the data frame
    """
    stringCols = df.select_dtypes(include=['object']).columns
    col_to_print = stringCols if cat_col == None else cat_col
    
    for column in col_to_print:
        print(df[column].value_counts())
        print('-'*50)


#function to print classification report and get confusion matrix in a proper format

def metrics_score(actual, predicted):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix    
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        xticklabels=['Not Canceled', 'Canceled'],
        yticklabels=['Not Canceled', 'Canceled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def metrics_score(actual, predicted, class_names_list):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True, fmt='.0f', 
                xticklabels=class_names_list, yticklabels=class_names_list)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def viz_missing_values(df):
    from datetime import date

    f, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))

    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(0), ax=ax[0],
                 color='darkorange', label = 'modified')
    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf),
                 ax=ax[0], color='dodgerblue', label = 'original')
    ax[0].set_title('Fill NaN with 0', fontsize=14)
    ax[0].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

    mean_drainage = df['drainage_volume'].mean()
    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(mean_drainage),
                 ax=ax[1], color='darkorange', label = 'modified')
    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf),
                 ax=ax[1], color='dodgerblue', label = 'original')
    ax[1].set_title(f'Fill NaN with Mean Value ({mean_drainage:.0f})', fontsize=14)
    ax[1].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

    sns.lineplot(x=df['date'], y=df['drainage_volume'].ffill(), ax=ax[2],
                 color='darkorange', label = 'modified')
    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[2],
                 color='dodgerblue', label = 'original')
    ax[2].set_title(f'FFill', fontsize=14)
    ax[2].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

    sns.lineplot(x=df['date'], y=df['drainage_volume'].interpolate(), ax=ax[3],
                 color='darkorange', label = 'modified')
    sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[3],
                 color='dodgerblue', label = 'original')
    ax[3].set_title(f'Interpolate', fontsize=14)
    ax[3].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

    for i in range(4):
        ax[i].set_xlim([date(2019, 5, 1), date(2019, 10, 1)])
        
    plt.tight_layout()
    plt.show()

    def precision_recall_at_k(model, k=10, threshold=3.5):
        from collections import defaultdict 
        
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        
        #Making predictions on the test data
        predictions=model.test([])
        
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set Precision to 0 when n_rec_k is 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set Recall to 0 when n_rel is 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
        #Mean of all the predicted precisions are calculated.
        precision = round((sum(prec for prec in precisions.values()) / len(precisions)),3)
        #Mean of all the predicted recalls are calculated.
        recall = round((sum(rec for rec in recalls.values()) / len(recalls)),3)
        
        accuracy.rmse(predictions)
        print('Precision: ', precision) #Command to print the overall precision
        print('Recall: ', recall) #Command to print the overall recall
        print('F_1 score: ', round((2*precision*recall)/(precision+recall),3)) # Formula to compute the F-1 score.


def explode1(pattern: str) -> Iterable[str]:
    """
    Expand the brace-delimited possibilities in a string.
    """
    seg_choices = []
    for segment in re.split(r"(\{.*?\})", pattern):
        if segment.startswith("{"):
            seg_choices.append(segment.strip("{}").split(","))
        else:
            seg_choices.append([segment])

    for parts in itertools.product(*seg_choices):
        yield "".join(parts)

def explode(pattern: str) -> Iterable[str]:
    """
    Expand the brace-delimited possibilities in a string.
    """
    import itertools
    import re

    seg_choices = (
        seg.strip('{}').split(',') if seg.startswith('{') else [seg]
        for seg in re.split(r'(\{.*?\})', pattern)
    )

    return (''.join(parts) for parts in itertools.product(*seg_choices))


# from plotly.offline import init_notebook_mode, plot, iplot
# import plotly.graph_objs as go
# init_notebook_mode(connected=True)

# data = df['bookRating'].value_counts().sort_index(ascending=False)
# trace = go.Bar(x = data.index,
#                text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
#                textposition = 'auto',
#                textfont = dict(color = '#000000'),
#                y = data.values,
#                )
# # Create layout
# layout = dict(title = 'Distribution Of {} book-ratings'.format(df.shape[0]),
#               xaxis = dict(title = 'Rating'),
#               yaxis = dict(title = 'Count'))
# # Create plot
# fig = go.Figure(data=[trace], layout=layout)
# iplot(fig)


if __name__ == '__main__':
    print(list(explode1("{Alice,Bob} ate a {banana,donut}.")))