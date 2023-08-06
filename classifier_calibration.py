from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
<<<<<<< HEAD
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
=======
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
>>>>>>> parent of 2f01f6c (tiral)

import numpy as np
import matplotlib.pyplot as plt

class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log(prob_true/(1-prob_true)) # transforming the datapoints to ln(x/(1-x))
        
        # use linearRegressor to make prob_pred
        self.regressor = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1))
    
    def calibrate(self, probabilities):
        # use sigmoid function (1 /(1+exp(-x))) to tranform datapoints back to original domain
        return 1/(1+np.exp(-self.regressor.predict(probabilities.reshape(-1,1)).flatten()))
    
    def _filter_out_of_domain(self, prob_pred, prob_true):
        filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
        return np.array(filtered)

class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds='clip')
        self.regressor.fit(prob_pred, prob_true)
        
    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)

class CalibratableModelMixin:
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__
        self.sigmoid_calibrator = None
        self.isotonic_calibrator = None
        self.calibrators = {
            'sigmoid': None,
            'isotonic': None,
        }
    
    def calibrate(self, X, y):
        predictions = self.predict(X)
        prob_true, prob_pred = calibration_curve(y, predictions, n_bins=10)
        self.calibrators['sigmoid'] = SigmoidCalibrator(prob_pred, prob_true)
        self.calibrators['isotonic'] = IsotonicCalibrator(prob_pred, prob_true)
    
    def calibrate_probabilities(self, probabilities, method='isotonic'):
        if method not in self.calibrators:
            raise ValueError("Methods has to be either 'sigmoid' or 'isotonic'")
        if self.calibrators[method] is None:
            raise ValueError('Run calibrate to fit the calibrators first')
        return self.calibrators[method].calibrate(probabilities)
    
    def predict_calibrated(self, X, method='isotonic'):
        return self.calibrate_probabilities(self.predict(X), method)
    
    def score(self, X, y):
        return self._get_accuracy(y, self.predict(X))
    
    def score_calibrated(self, X, y, method='isotonic'):
        return self._get_accuracy(y, self.predict_calibrated(X, method))
    
    def _get_accuracy(self, y, preds):
        return np.mean(np.equal(y.astype(bool), preds >= 0.5))

<<<<<<< HEAD
=======
def plot_calibration_curve(y, probs, title):
    brier_score = brier_score_loss(y, probs)
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(
        prob_pred,
        prob_true,
        marker='.',
        color='orange')
    plt.title(f'{title}\nBrier score: {brier_score:.3f}')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    return prob_true, prob_pred

def plot_fitted_calibrator(prob_true, prob_pred, prob_calibrated, title=None):
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker=".", color="orange")
    plt.plot(prob_pred, prob_calibrated, color="red")
    plt.title(title)
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")


def plot_calibration_details_for_models(models, X, y, 
                                        calibrated=False, method="isotonic"):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model in models:
        name = model.name
        probabilities = (
            model.predict(X)
            if not calibrated
            else model.predict_calibrated(X, method=method)
        )
        prob_true, prob_pred = calibration_curve(y, probabilities, n_bins=10)
        brier_score = brier_score_loss(y, probabilities)

        ax1.plot(
            prob_pred,
            prob_true,
            marker=".",
            label=f"{name} (BS={round(brier_score, 3)})",
        )

        ax2.hist(
            probabilities, range=(0, 1), bins=10, label=name, histtype="step", lw=2
        )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title("Calibration plots")
    ax1.grid()

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax2.grid()

>>>>>>> parent of 2f01f6c (tiral)
class NNModel(CalibratableModelMixin):
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)