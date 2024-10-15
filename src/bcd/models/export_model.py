import joblib
from sklearn.base import BaseEstimator

from bcd.utils.paths import models_dir


def export_model(model: BaseEstimator, scoring: str, score: float) -> None:
    """
    Exports a trained model to a file if the model achieves a 
    satisfactory performance score.
    
    Parameters
    ----------
    model : BaseEstimator
        The trained machine learning model that will be exported. 
        This should be an instance of a class that inherits from
        scikit-learn's BaseEstimator.
    score : float
        The performance score of the model (e.g., accuracy, F1 score). 
        It will be displayed when exporting the model and could be used 
        to track the model's effectiveness.
    """
    
    print(f'Model score: {score}')
    models_DIR = models_dir(f'best_model_for_{scoring}.pkl')
    joblib.dump(model, models_DIR)