import numpy as np

class Stage2XGBoost:
    """
    Wrapper for Stage-2 XGBoost classifier.
    """
    def __init__(self, config=None, model=None):
        import xgboost as xgb
        self.config = config or {}
        if model is not None:
            self.model = model
        else:
            self.model = xgb.XGBClassifier(
                objective=self.config.get('objective', 'binary:logistic'),
                eval_metric=self.config.get('eval_metric', 'logloss'),
                random_state=self.config.get('random_seed', 42),
                tree_method='hist',
                n_jobs=1
            )
            
    def set_params(self, **params):
        self.model.set_params(**params)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
            
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)[:, 1]
        return (probas > threshold).astype(int)
        
    def save(self, path):
        self.model.save_model(path)
        
    @classmethod
    def load(cls, path, config=None):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(path)
        return cls(config, model)
