import pandas as pd
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

from bcd.models.export_model import export_model


class TrainModel:
    """
    A class to train machine learning models using evolutionary 
    hyperparameter optimization.

    This class utilizes genetic algorithms (via GASearchCV) to 
    perform hyperparameter optimization for predefined models. 
    Currently supports the XGBoost classifier.

    Attributes
    ----------
    estimators : dict
        Dictionary of model estimators to be trained.
    param_grids : dict
        Dictionary of hyperparameter search spaces corresponding 
        to each model.
    """

    def __init__(self) -> None:
        """
        Initializes the TrainModel class with the estimators and their 
        corresponding hyperparameter grids.

        Currently, only the XGBoost classifier is defined in the estimators dictionary.
        """

        self.estimators = {
            'XGBOOST': XGBClassifier()
        }

        self.param_grids = {
            'XGBOOST': {
                'n_estimators': Integer(40, 400),
                'learning_rate': Continuous(0.01, 0.2),
                'max_depth': Integer(3, 10),
                'subsample': Continuous(0.5, 1),
                'colsample_bytree': Continuous(0.5, 1),
                'gamma': Integer(0, 1),
                'reg_lambda': Continuous(0, 10),
                'reg_alpha': Continuous(0, 10),
                'min_child_weight': Continuous(0, 10)
            }
        }

    def genopt_training(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = 'accuracy',
    ) -> BaseEstimator:
        """
        Trains models using genetic algorithm-based hyperparameter 
        optimization and exports the best model.

        This method performs an evolutionary search over hyperparameters 
        using GASearchCV for each estimator defined in the `estimators` 
        dictionary. After evaluating the performance of all models, it 
        exports the best one based on accuracy score.

        Parameters
        ----------
        X : pd.DataFrame
            The feature data to train the models on.
        y : pd.Series
            The target labels corresponding to the feature data.
        """

        best_score = 0
        best_model = None

        for name, estimator in self.estimators.items():
            print(f"Performing evolutionary optimization for {name}...")

            evolved_estimator = GASearchCV(
                estimator=estimator,
                cv=3,
                scoring=scoring,
                param_grid=self.param_grids[name],
                n_jobs=-1,
                verbose=True,
                population_size=15,
                generations=8
            ).fit(X, y)

            score = evolved_estimator.best_score_

            print(f"Best {name} model has {scoring} score of {score:.2f}")

            if score > best_score:
                best_score = score
                best_model = evolved_estimator.best_estimator_

        # Check if a best model was found, and export it
        if best_model is not None:
            print(
                f"Exporting best model with {scoring} score of {best_score:.2f}")
            export_model(best_model, scoring=scoring, score=best_score)
            return evolved_estimator
        else:
            raise ValueError(
                "No suitable model was found during optimization.")
