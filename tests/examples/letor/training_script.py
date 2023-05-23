"""Training script for a learning-to-rank task"""
import xgboost

dtrain = xgboost.DMatrix("mq2008.train")
dvalid = xgboost.DMatrix("mq2008.vali")
dtest = xgboost.DMatrix("mq2008.test")

params = {
    "objective": "rank:ndcg",
    "eta": 0.01,
    "gamma": 1.0,
    "min_child_weight": 0.1,
    "max_depth": 8,
    "silent": 1,
    "eval_metric": "ndcg",
}
# num_boost_round=713 was chosen using early stopping on validation set mq2008.vali
xgb_model = xgboost.train(
    params,
    dtrain,
    num_boost_round=713,
    evals=[(dtrain, "train"), (dtest, "test"), (dvalid, "validation")],
    verbose_eval=True,
)
xgb_model.save_model("mq2008.model")
