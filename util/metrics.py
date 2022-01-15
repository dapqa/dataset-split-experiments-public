from numba import jit, types
from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(expected, actual):
    return mean_squared_error(expected, actual, squared=False)


@jit(
    types.float64(
        types.Array(types.int64, 2, 'C'),
        types.Array(types.float64, 1, 'C'),
        types.Array(types.int64, 2, 'C'),
        types.int64
    ),
    nopython=True, fastmath=True, cache=True
)
def _ngdc_at_k_numba(X_true, rating_true, X_pred, k):
    true_min_max_ranks = dict()

    idcg_values = np.zeros(k + 1)
    for i in range(1, k + 1):
        idcg_values[i] = idcg_values[i - 1] + 1 / np.log2(i + 1)

    prev_true_user_id = X_true[0][0]
    prev_true_rating = rating_true[0]
    min_rank = 1
    max_rank = 0
    item_ids_with_this_rating = []
    for i in range(len(rating_true)):
        true_user_id = X_true[i][0]
        true_item_id = X_true[i][1]
        true_rating = rating_true[i]

        if true_user_id != prev_true_user_id:
            for iid in item_ids_with_this_rating:
                true_min_max_ranks[(prev_true_user_id, iid)] = (min_rank, max_rank)
            item_ids_with_this_rating.clear()
            item_ids_with_this_rating.append(true_item_id)

            min_rank = 1
            max_rank = 1
            prev_true_user_id = true_user_id
            prev_true_rating = true_rating
        elif true_rating != prev_true_rating:
            for iid in item_ids_with_this_rating:
                true_min_max_ranks[(prev_true_user_id, iid)] = (min_rank, max_rank)
            item_ids_with_this_rating.clear()
            item_ids_with_this_rating.append(true_item_id)

            min_rank = max_rank + 1
            max_rank = min_rank
            prev_true_rating = true_rating
        else:
            max_rank += 1
            item_ids_with_this_rating.append(true_item_id)

    for iid in item_ids_with_this_rating:
        true_min_max_ranks[(prev_true_user_id, iid)] = (min_rank, max_rank)

    cur_k = 0
    prev_pred_user_id = X_pred[0][0]
    dcg_at_k = 0
    res = 0
    user_count = 1
    for i in range(len(X_pred)):
        pred_user_id = X_pred[i][0]
        pred_item_id = X_pred[i][1]

        if pred_user_id != prev_pred_user_id:
            idcg_at_k = idcg_values[cur_k]
            if idcg_at_k > 0:
                res += dcg_at_k / idcg_at_k

            user_count += 1
            dcg_at_k = 0
            cur_k = 0
            prev_pred_user_id = pred_user_id

        if cur_k < k:
            cur_k += 1

            min_true_rank, max_true_rank = true_min_max_ranks.get((pred_user_id, pred_item_id), (0, 0))
            if min_true_rank <= cur_k <= max_true_rank:
                dcg_at_k += 1 / np.log2(cur_k + 1)

    idcg_at_k = idcg_values[cur_k]
    if idcg_at_k > 0:
        res += dcg_at_k / idcg_at_k

    return res / user_count


def ndcg_at_k(X, y_expected, y_actual, k=10, X_actual=None):
    rating_true = np.c_[X, y_expected]

    if X_actual is None:
        rating_pred = np.c_[X, y_actual]
    else:
        rating_pred = np.c_[X_actual, y_actual]

    rating_true = rating_true[np.lexsort((rating_true[:, 2], rating_true[:, 0]))][::-1]
    rating_pred = rating_pred[np.lexsort((rating_pred[:, 2], rating_pred[:, 0]))][::-1]

    return _ngdc_at_k_numba(
        X_true=rating_true[:, 0:2].astype('int64'),
        rating_true=rating_true[:, 2].astype('float64'),
        X_pred=rating_pred[:, 0:2].astype('int64'),
        k=k
    )
