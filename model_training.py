from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def optimize_hyperparameters(classifier, X, y, parameters, random=False, n_iter=200, scoring=None):
    if random:
        clf_cv = RandomizedSearchCV(classifier, param_distributions=parameters, scoring=scoring, n_iter=n_iter, cv=5,
                                    refit=True, verbose=2)
    else:
        clf_cv = GridSearchCV(classifier, parameters, scoring=scoring, cv=5, refit=True)

    clf_cv.fit(X, y)
    print("---Optimizing---")
    print(f"Best parameters: {clf_cv.best_params_}")
    print(f"Best score: {clf_cv.best_score_}")
    print("----------------")
    return clf_cv.best_estimator_
