import pandas as pd
import os
import gc
import shap
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from scipy.stats import sem
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, RandomizedSearchCV, LeaveOneOut, cross_val_score
import joblib


def RandomGroupKFold_split(groups, n, seed=0):
    """
    Random analogue of sklearn.model_selection.GroupKFold.split.

    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result

EPS = 1e-3
# some material taken from
# github.com/manujosephv/interpretability_blog/blob/master/census_income_interpretability.ipynb

class oversampled_Kfold():
    def __init__(self, n_splits, random_state, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y):
        splits = np.array_split(np.random.choice(len(X), len(X),replace=False), self.n_splits)
        train, test = [], []
        for repeat in range(self.n_repeats):
            for fold in range(len(splits)):
                train_splits = splits[:]
                test_splits = train_splits.pop(fold)
                ros = RandomOverSampler(random_state=self.random_state)
                train_idx = np.concatenate(train_splits)
                Xidx_r, y_r = ros.fit_resample(train_idx.reshape(-1, 1), y.values[train_idx,0])
                train.append(Xidx_r.flatten())
                test.append(test_splits)
        return list(zip(train, test))


class RFShap(object):
    def __init__(self, model_dir=None, exclude_vars=None, outcome_var=None, output_dir=None, random_seed=42, group_var=None,
                 class_='RF', type_='reg', balanced='balanced', trn_tst_split=0.6, k_cv='k_fold', k=5, config=None, num_trials=15):

        """
        :param model_dir: this is for pre_trained model loading
        :param exclude_vars: list the variables to exclude from the analysis (if any)
        :param outcome_var: name the outcome variable
        :param output_dir: name the directory to store results
        :param random_seed: set random seed
        :param class_: either 'RF' for random forest, or 'lin' for linear/logistic
        :param type_:  either 'reg' or 'cls' for regression or classification
        :param balanced:  either 'balanced' or 'unbalanced' for imblearn or sklearn respectively
        :param group_var: either None or a string corresponding with the dataframe column name
        :param k_cv: uses k_fold training across entire dataset.
        :param k: if k_cv= 'split', 'k_fold'
        :param config: specifies settings for the classifiers/random forests (otherwise will use defaults)
        Config can be used to specify a list of hyperparameters
        """
        assert class_ in ['RF', 'lin'], 'Class not recognised - choose reg or lin.'
        assert type_ in ['reg', 'cls'], 'Type not recognised - choose reg or cls.'
        assert balanced in ['balanced', None], 'Balanced must be balanced or unbalanced'
        assert outcome_var is not None, 'Pick an outcome variable!'
        assert k_cv in ['k_fold', 'split'], "For k_cv Choose from 'k_fold', 'split'"

        self.group_var = group_var
        self.group_data = None  # this will be populated if groups are used
        self.model_dir = model_dir
        self.exclude_vars = exclude_vars
        self.outcome_var = outcome_var
        self.output_dir = output_dir
        self.seed = random_seed
        self.class_ = class_
        self.type_ = type_
        self.balanced = balanced
        self.trn_tst_split = trn_tst_split
        self.k_cv = k_cv
        self.k = k
        self.dataset = None
        self.model = None
        self.cat_list = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        # TODO: set up config as empty dict i.e. config = {} to avoid needing conditional statements in make_model()
        self.config = config
        self.n_categories = None
        self.shap_interaction_vals = None

        if self.model_dir is not None:
            self.model = self.load_model(model_dir_=self.model_dir)

        if self.group_var is not None and self.balanced:
            print('NOTE: <BALANCED> GROUP K FOLD IS NOT AVAILABLE, DEFAULTING TO GROUP K-FOLD')


    def munch(self, dataset):
        """
        :param dataset:
        :return: self.dataset, self.X (predictors), self.y (targets), self.X_train, self.X_test, self.y_train, self.y_test
        """

        print('Preparing dataset...')
        self.dataset = dataset
        if self.group_var is not None:
            self.group_data = self.dataset[self.group_var]
        if self.exclude_vars is not None:
            self.dataset = dataset.drop(columns=self.exclude_vars)
        self.dataset = self.dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        print('Check if continuous or categorical variables: ')
        self.cat_list = self.find_categorial(self.dataset)
        print(self.cat_list)
        self.y = self.dataset[[self.outcome_var]]
        self.n_categories = len(np.unique(self.y))
        self.X = self.dataset.drop(columns=self.outcome_var)


        if self.class_ == 'lin':
            self.X = (self.X-self.X.mean())/(self.X.std() + EPS)

        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

        return self.dataset,  self.X,  self.y, self.X_train, self.X_test, self.y_train, self.y_test, self.group_data

    def load_model(self, model_dir_=None):
        model_dir = model_dir_

        if model_dir is None:
            model_dir = self.model_dir
        else:
            self.model_dir = model_dir_

        try:
            self.model = joblib.load(model_dir)
        except:
            print('Either no model dir specified, or model dir is incorrect (i.e. no /.../model.sav found).')

        return self.model

    def save_model(self, model_dir_=None):

        model_dir = model_dir_
        if model_dir is None:
            model_dir = self.model_dir
        else:
            self.model_dir = model_dir_
        try:
            joblib.dump(self.model, model_dir)
        except:
            print('Check model dir... had problems saving /.../model.sav.')


    def make_model(self, config=None):
        """
        :param config : model parameters
        :return: self.model
        """

        if config != None:
            self.config = config

        print('Creating fresh model...')

        if self.class_ == 'RF':
            if self.type_ == 'reg':
                if self.balanced == 'balanced':
                    print('WARNING: balanced regressor not applicable')
                    self.model = RandomForestRegressor(**config) if config != None else RandomForestRegressor(random_state=self.seed)
                elif self.balanced == None:
                    self.model = RandomForestRegressor(**config) if config != None else RandomForestRegressor(random_state=self.seed)
            elif self.type_ == 'cls':
                if self.balanced == 'balanced':
                    self.model = BalancedRandomForestClassifier(**config) if config != None else BalancedRandomForestClassifier(random_state=self.seed)
                elif self.balanced == None:
                    self.model = RandomForestClassifier(**config) if config != None else RandomForestClassifier(random_state=self.seed)
        elif self.class_ == 'lin':
            if self.type_ == 'reg':
                if self.balanced == 'balanced':
                    print('WARNING: balanced regressor not applicable')
                    self.model = LinearRegression(**config) if config != None else LinearRegression()
                elif self.balanced == None:
                    self.model = LinearRegression(**config) if config != None else LinearRegression()
            elif self.type_ == 'cls':
                if self.balanced == 'balanced':
                    self.model = LogisticRegression(**config) if config != None else LogisticRegression()
                    self.model.class_weight = self.balanced
                elif self.balanced == None:
                    self.model = LogisticRegression(**config) if config != None else LogisticRegression()
                    self.model.class_weight = None
        elif self.class_ == 'svm':
            assert self.type_ == 'cls', print('If using SVM, make sure you have a classification problem. i.e. set type_="cls"')
            self.model = SVC(**config) if config != None else SVC(kernel='rbf')

        print('Created: ', self.model)
        return self.model


    def _split_data(self):
        if self.trn_tst_split < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                test_size=self.trn_tst_split, random_state=self.seed)
        elif self.trn_tst_split == 1.0:
            X_train = self.X
            X_test = self.X
            y_train = self.y
            y_test = self.y
        return X_train, X_test, y_train, y_test

    def train_test(self, plot=False):

        """
        :param plot: whether to plot the auroc with k-fold training (default is false)
        :return: trained model and performance report
        """

        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'
        report = None
        if self.k_cv == 'k_fold':
            report, conf_mat = self._train_eval_kloo(plot=plot)

        elif self.k_cv == 'split':
            self.model.fit(self.X_train, self.y_train.values.ravel())
            report, conf_mat = self._eval_split()


        joblib.dump(self.model, os.path.join(self.output_dir,  str(self.type_) + '_' + str(self.class_) +
                                          '_model.joblib'))
        return self.model, report

    def _eval_split(self):

        """
        :return: performance report
        """
        assert self.model != None, 'model cannot be NoneType! Make sure it has been defined.'

        preds = self.model.predict(self.X_test)
        df_report = None

        if self.type_ == 'cls':
            print("Accuracy: %s%%" % (100 * metrics.accuracy_score(self.y_test, preds)))
            conf_mat = np.asarray(metrics.confusion_matrix(self.y_test.values.ravel(), preds))
            np.savetxt(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' + str(self.class_) +
                                    '_simple_test_conf_mat.txt'), conf_mat)
            report = classification_report_imbalanced(self.y_test, preds)
            df_report = self.imblearn_rep_to_df(report)
            df_report['mcc'] = metrics.matthews_corrcoef(self.y_test, preds)
            df_report.to_csv(os.path.join(self.output_dir,  str(self.type_) + '_' + str(self.class_) +
                                          '_simple_test_classification_report.csv'), index=False)
            print(df_report)

        elif self.type_ == 'reg':
            expl_var = metrics.explained_variance_score(self.y_test, preds)
            mae = metrics.mean_absolute_error(self.y_test, preds)
            mse = metrics.mean_squared_error(self.y_test, preds)
            try:
                msle = metrics.mean_squared_log_error(self.y_test, preds)
            except:
                msle = 0
            med_ae = metrics.median_absolute_error(self.y_test, preds)
            r2 = metrics.r2_score(self.y_test, preds)
            cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
            df_report = pd.DataFrame([expl_var, mae, mse, msle, med_ae, r2]).T
            df_report.columns = cols
            df_report.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                          '_simple_test_regression_report.csv'), index=False)

        return df_report



    def _train_eval_kloo(self, plot=False):
        # adapted from https://scikit-learn.org/

        def objective(trial):
            self.model = None
            n_estimators = trial.suggest_int("n_estimators", 50, 1000)
            max_depth = trial.suggest_int("max_depth", 30, 600, log=True)
            trial_config = {'n_estimators': n_estimators, 'max_depth': max_depth}
            self.model = self.make_model(trial_config)

            scoring = 'balanced_accuracy' if self.type_ == 'cls' else 'explained_variance'
            score = cross_val_score(self.model, X_train, y_train.values.ravel(), n_jobs=3, cv=self.k, scoring=scoring)
            score = score.mean()
            return score

        if self.type_ == 'cls':
            if self.k_cv == 'k_fold':
                if self.balanced:
                    kf = oversampled_Kfold(n_splits=self.k, shuffle=True, random_state=self.seed)
                    kfsplit = kf.split(self.X, self.y)
                    if self.group_var is not None:
                        kfsplit = RandomGroupKFold_split(groups=self.group_data.values, n=self.k, seed=self.seed)
                else:
                    kf = KFold(n_splits=self.k, random_state=self.seed, shuffle=True,)
                    kfsplit = kf.split(self.X)
                    if self.group_var is not None:
                        kfsplit = RandomGroupKFold_split(groups=self.group_data.values, n=self.k, seed=self.seed)
                test_accs = []
                tprs = []
                aucs = []
                tprs_2 = []
                cms = []
                fprs = []
                precisions = []
                rocaucscores = []
                mccs = []
                recalls = []
                mean_fpr = np.linspace(0, 1, 100)
                # report = {}

                y_probs = []
                y_preds = []
                y_GTs = []

                results = pd.DataFrame()
                i = 0
                f = 0
                if self.n_categories <= 2:
                    fig, ax = plt.subplots(figsize=(14, 7))

                for train_index, test_index in kfsplit:
                    f += 1
                    print('Training fold: ', f)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]


                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    preds = self.model.predict(X_test)
                    probs = self.model.predict_proba(X_test)

                    y_probs.append(probs)
                    y_preds.append(preds)
                    y_GTs.append(y_test)

                    test_accs.append(100 * metrics.accuracy_score(y_test, preds))
                    rep = classification_report_imbalanced(y_test, preds)
                    # rep = self.unravel_report(rep)
                    # report = self.combine_dicts(report, rep)
                    if self.n_categories <= 2:
                        probs_ = probs[:, 1]
                    else:
                        probs_ = probs
                    rocaucscores.append(roc_auc_score(y_test, probs, average='macro', multi_class='ovr'))
                    rep = self.imblearn_rep_to_df(rep)
                    rep['fold'] = f
                    results = pd.concat([results, rep])

                    if self.n_categories <= 2:
                        mccs.append(metrics.matthews_corrcoef(y_test, preds))
                        fpr, tpr, _ = metrics.roc_curve(y_test, probs[:, 0])
                        tprs_2.append(tpr)
                        fprs.append(fpr)
                        precisions.append(metrics.precision_score(y_test, preds, average=None))
                        recalls.append(metrics.recall_score(y_test, preds, average=None))
                        viz = metrics.plot_roc_curve(self.model, X_test, y_test,
                                             name='ROC fold {}'.format(i),
                                             alpha=0.3, lw=1, ax=ax)
                        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        aucs.append(viz.roc_auc)
                    i += 1


                if self.n_categories <= 2:
                    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                            label='Chance', alpha=.8)

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = metrics.auc(mean_fpr, mean_tpr)
                    std_auc = np.std(aucs)
                    ax.plot(mean_fpr, mean_tpr, color='b',
                            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                            lw=2, alpha=.8)

                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                    label=r'$\pm$ 1 std. dev.')

                    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                           title="Receiver operating characteristic example")
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' +'_k_fold_AUROC.png'))
                    if plot:
                        plt.show()
                    plt.close()

                print('accuracies', np.asarray(test_accs).mean())
                print('roc score', np.asarray(rocaucscores).mean())
                print(results)


                if self.n_categories <= 2:
                    mccs = pd.DataFrame(mccs)
                    print('Matthews Correlation Coefficients: ', mccs)
                    mccs = mccs.append(mccs.mean(), ignore_index=True)
                    mccs = mccs.append(mccs.std()/np.sqrt(self.k), ignore_index=True)
                    print('Last two rows of mccs_k_fold.csv are MEAN and Standard Err. respectively!')
                    mccs.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                                  '_mccs_k_fold.csv'), index=False)
                # results = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in report.items()]))
                self.save_kfold_summary(results)

                y_preds = np.concatenate(y_preds)
                y_GTs = np.concatenate(y_GTs)
                cms = metrics.confusion_matrix(y_GTs, y_preds, normalize=None)
                np.savetxt(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' + str(
                    self.class_) + '_kcv_unnormalized_conf_mat.txt'), cms)
                bas_n = balanced_accuracy_score(y_GTs, y_preds, adjusted=True)
                bas = balanced_accuracy_score(y_GTs, y_preds, adjusted=False)
                print('balanced accuracy adjusted, unadjusted:', bas_n, bas)
                bas_df = pd.DataFrame()
                bas_df['adjusted_balanced_acc'] = np.array([bas_n])
                bas_df['unadjusted_balanced_acc'] = np.array([bas])
                bas_df.to_csv(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' + str(
                    self.class_) + 'kcv_balanced_acc_scores.csv'))


        elif self.type_ == 'reg':
            cms = None
            if self.k_cv == 'k_fold':
                kf = KFold(n_splits=self.k, random_state=self.seed, shuffle=True,)
                kfsplit = kf.split(self.X)
                if self.group_var is not None:
                    kfsplit = RandomGroupKFold_split(groups=self.group_data.values,  n=self.k, seed=self.seed)
                expl_vars = []
                maes = []
                mses = []
                msles = []
                r2s = []
                med_aes = []
                f = 0
                for train_index, test_index in kfsplit:
                    f += 1
                    print('Training fold: ', f)
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                    self.model = self.make_model(self.config)
                    self.model.fit(X_train, y_train.values.ravel())
                    preds = self.model.predict(X_test)
                    expl_vars.append(metrics.explained_variance_score(y_test, preds))
                    maes.append(metrics.mean_absolute_error(y_test, preds))
                    mses.append(metrics.mean_squared_error(y_test, preds))
                    try:
                        msles.append(metrics.mean_squared_log_error(y_test, preds))
                    except:
                        pass
                    med_aes.append(metrics.median_absolute_error(y_test, preds))
                    r2s.append(metrics.r2_score(y_test, preds))

                expl_vars = np.asarray(expl_vars)
                maes = np.asarray(maes)
                mses = np.asarray(mses)
                msles = np.asarray(msles)
                r2s = np.asarray(r2s)
                med_aes = np.asarray(med_aes)

                results = pd.DataFrame([expl_vars, maes, mses, msles, med_aes, r2s]).T
                cols = ['expl_var', 'mae', 'mse', 'msle', 'med_ae', 'r2']
                results.columns = cols
                results = results.append(results.mean(), ignore_index=True)
                print('Last row is the mean across columns.')
                results.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                     '_k_fold_test_regression_report.csv'), index=False)



        return results, cms




    def run_shap_explainer(self, model, check_additivity=True, fraction_data=1.0):

        """
        :param model: give it a trained model
        :return: explainer and shapley values
        """
        assert model is not None, 'Feed me a model : ]'

        print('Using ', fraction_data, ' of the data to run Shap.')
        # indexes = np.random.choice(np.arange(0, len(self.X)), int(len(self.X)*fraction_data))
        # X_test = self.X.iloc[indexes]
        X_test = self.X

        model_output = 'margin' if self.type_ == 'reg' else 'raw'   # if using RF classifier, raw is log odds
        print('Running Shap Explainer.')
        explainer = shap_vals = None

        # 'tree_path_dependent' does the 'true to data' approach rather than the interventional approach which is
        # true to the model, see https://arxiv.org/abs/2006.16234
        # (Hugh Chen, Joseph D. Janizek, Scott Lundberg, Su-In Lee 2020)
        # note that the interventioanl option requires a 'background dataset'

        if self.class_ == 'RF':
            print('Running Explainer')
            explainer = shap.TreeExplainer(model=model, model_output=model_output, feature_perturbation="tree_path_dependent", check_additivity=check_additivity)
            joblib.dump(explainer, os.path.join(self.output_dir, self.outcome_var + "_tree_explainer.sav"))
            print('Generating Shap values')
            shap_vals = explainer.shap_values(X_test, check_additivity=check_additivity)
            joblib.dump(shap_vals, os.path.join(self.output_dir, self.outcome_var + "_tree_explainer_shap_values.sav"))
        elif self.class_ == 'lin':
            print('Running Explainer')
            explainer = shap.LinearExplainer(model, X_test, check_additivity=check_additivity)
            joblib.dump(explainer, os.path.join(self.output_dir, self.outcome_var + '_linear_explainer.sav'))
            print('Generating Shap values')
            shap_vals = explainer.shap_values(X_test, check_additivity=check_additivity)
            joblib.dump(shap_vals, os.path.join(self.output_dir, self.outcome_var + '_linear_explainer_shap_values.sav'))

        if self.type_ == 'reg':
            inds = np.flip(np.argsort(np.abs(shap_vals).mean(0)))
            print(shap_vals.shape, inds)
            sorted_vals = np.abs(shap_vals).mean(0)[(inds)]

        else:
            inds = np.flip(np.argsort(np.abs(shap_vals).mean(0).mean(0)))
            sorted_vals = np.abs(shap_vals).mean(0).mean(0)[(inds)]

        imps = pd.DataFrame(sorted_vals).T
        imps.columns = self.X.columns[inds]
        ims_dir = os.path.join(self.output_dir, self.outcome_var + '_importances.csv')
        imps.to_csv(ims_dir, index=False)

        return explainer, shap_vals

    def shap_bootstrap(self, model=None, retrain=False, n_bootstraps=1000, n_samples=100, class_ind=0):
        """

        :param model: desired model  (if None then will use model that you trained earlier within this class)
        :param retrain: this retrains the model for each bootstrap
        :param n_bootstraps: number of boostraps to undertake
        :param n_samples: sample size used in each bootstrap
        :param class_ind: for multiple classes, if you want to look at a particular class
        :return: shapley values for all bootstraps, and report for mean and standard error
        """

        assert n_samples < len(self.X_test), 'n_samples cannot be longer than the test set! Reduce n_samples.'

        if model == None:
            model = self.model
        shap_vals_bootstraps = []

        if self.k_cv == 'split':
            X_test_bootstrap = self.X
        elif self.k_cv == 'k_fold':
            X_test_bootstrap = self.X
        indices = np.arange(0, len(X_test_bootstrap))
        b = 0
        for _ in range(n_bootstraps):
            b += 1
            if b % 20 == 0:
                print('Bootstrap number: ', b)
            if retrain:
                model = self.make_model(self.config)
                model, _ = self.train_test(plot=False)
            sample_inds = np.random.choice(indices, n_samples)
            X_test_sample = X_test_bootstrap.iloc[sample_inds]
            if self.class_ == 'RF':
                model_output = 'margin' if self.type_ == 'reg' else 'raw'
                explainer = shap.TreeExplainer(model=model, model_output=model_output, feature_perturbation="tree_path_dependent")
                shap_vals = explainer.shap_values(X_test_sample)

                if self.type_ == 'cls':
                    shap_vals = shap_vals[class_ind]

            elif self.class_ == 'lin':
                explainer = shap.LinearExplainer(model, X_test_sample)
                shap_vals = explainer.shap_values(X_test_sample)
            elif self.class_ == 'svm':
                print('Not implemented bootstrapped shap with svm yet')

            shap_vals_bootstraps.append(shap_vals)

        shap_vals_bootstraps = np.asarray(shap_vals_bootstraps)
        abs_shaps = np.abs(shap_vals_bootstraps)
        mean_shaps = np.mean(abs_shaps, 1)
        se_bootstraps = sem(mean_shaps, 0)
        mean_bootstraps = np.mean(mean_shaps, 0)
        sorted_inds = np.argsort(mean_bootstraps)

        plt.figure(figsize=(7, 10))
        plt.barh(X_test_bootstrap.columns[sorted_inds][-20:], mean_bootstraps[sorted_inds][-20:], xerr=se_bootstraps[sorted_inds][-20:] * 1.96)
        plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, self.outcome_var + 'bootstrap_shap.png'))
        plt.show()

        bootstrap_results = pd.DataFrame([mean_bootstraps, se_bootstraps])
        bootstrap_results.columns = self.X_test.columns
        bootstrap_results.to_csv(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) +
                                                '_' + str(self.class_) + '_shapley_importances_bootstrap_results.csv'), index=False)
        return shap_vals_bootstraps, bootstrap_results


    def shap_plot(self, explainer=None, shap_vals=None, specific_var=None, interactions=False,
                  interaction_vars=None, classwise=True, val_summary=False, force_plot=False,
                  class_name=None, class_ind=1, num_display=20):

        """
        :param explainer: explainer
        :param shap_vals: vals derived from running the explainer
        :param specific_var: if desired, run the individual feature plots
        :param interaction_var: which desired var to plot as interacting with 'specific var'
        :param class ind: when plotting classifier results, pick class index to plot with
        :return shap_interaction_vals: these are expensive to compute, so only want to do so once!
        """
        interaction_var = None
        if interaction_vars is not None:
            if len(interaction_vars) == 0:
                raise Exception('If there are no interactions vars, specify None, not empty list []')
            if len(interaction_vars) > 2:
                raise Exception('Interaction vars list cannot be greater than 2.')

        if class_name == None:
            class_name = class_ind

        def plot_interactions(data, expl=None, vars_=None, class_index=1):
            if self.shap_interaction_vals is None:
                if self.type_ == 'cls':
                    self.shap_interaction_vals = expl.shap_interaction_values(data)[class_index]
                elif self.type_ == 'reg':
                    self.shap_interaction_vals = expl.shap_interaction_values(data)

            tmp = np.abs(self.shap_interaction_vals).sum(0)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 0
            inds = np.argsort(-tmp.sum(0))[:50]
            tmp2 = tmp[inds, :][:, inds]
            plt.figure(figsize=(12, 12))
            plt.imshow(tmp2)
            plt.yticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="right")
            plt.xticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="left")
            plt.gca().xaxis.tick_top()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' +
                        str(self.class_) + '_interaction_matrix_{}.png'.format(class_index)))
            plt.show()
            plt.close()

            if vars_ != None:
                shap.dependence_plot(
                    vars_,
                    self.shap_interaction_vals,
                    data, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' +
                                         str(self.class_) + '_interaction_{}_{}_{}.png'.format(vars_[0], vars_[1], class_index)))
                plt.show()
                plt.close()

        if self.k_cv == 'split':
            X_test_plot = self.X
        elif self.k_cv == 'k_fold':
            X_test_plot = self.X

        if self.type_ == 'cls':
            if interactions:
                plot_interactions(X_test_plot, explainer, interaction_vars, class_ind)
            if val_summary:
                if classwise or (self.class_ == 'lin'):
                    shap.summary_plot(shap_values=shap_vals, features=X_test_plot, max_display=num_display, plot_type='bar', show=False)
                    plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                             str(self.class_) + '_' + str(num_display) +'_shap_val_summary_{}.png'))
                    plt.show()
                    plt.close()
                else:
                    shap.summary_plot(shap_values=shap_vals[class_ind], features=X_test_plot, max_display=num_display, plot_type='bar', show=False)
                    plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, self.outcome_var + '_' + str(self.type_) + '_' +
                                             str(self.class_) + '_' + str(num_display) +'_shap_val_summary.png'))
                    plt.show()
                    plt.close()


            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals[class_ind], features=X_test_plot, max_display=num_display, plot_type='dot',
                              show=False)
            elif self.class_ == 'lin':
                shap.summary_plot(shap_values=shap_vals, features=X_test_plot, max_display=num_display, plot_type='dot',
                              show=False)
            elif self.class_ == 'svm':
                print('not implemented shap for svm yet')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + '_' + str(class_name) + '_' + str(num_display) +'_shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                     shap_values=shap_vals[class_ind], features=X_test_plot, show=False)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                             shap_values=shap_vals, features=X_test_plot, show=False)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_'
                                         + str(self.class_) + '_' + str(num_display) +'_shap_interaction_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()

        elif self.type_ == 'reg':
            if interactions:
                plot_interactions(X_test_plot, explainer, interaction_vars, class_ind)

            shap.summary_plot(shap_values=shap_vals, features=X_test_plot, max_display=num_display, plot_type='bar',
                              show=False)
            plt.xlabel('mean(|SHAP value|) (impact on output magnitude)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + '_' + str(num_display) +'_shap_val_summary.png'))
            plt.show()
            plt.close()

            if self.class_ == 'RF':
                shap.summary_plot(shap_values=shap_vals, features=X_test_plot, max_display=num_display, plot_type='dot',
                                  show=False)
            else:
                shap.summary_plot(shap_values=shap_vals, features=X_test_plot, max_display=num_display, plot_type='dot',
                          show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                     str(self.class_) + '_' + str(num_display) +'_shap_effects_summary.png'))
            plt.show()
            plt.close()

            if specific_var is not None:
                if self.class_ == 'RF':
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals, features=X_test_plot, show=False)
                else:
                    shap.dependence_plot(specific_var, interaction_index=interaction_var,
                                         shap_values=shap_vals, features=X_test_plot, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, self.outcome_var +'_' + str(self.type_) + '_' +
                                         str(self.class_) + '_' + str(num_display) +'_shap_interaction_summary_{}.png'.format(specific_var)))
                plt.show()
                plt.close()

        # visualize the training set predictions
        f = os.path.join(self.output_dir,
                         self.outcome_var + '_' + str(self.type_) + '_' + str(self.class_) + 'shap_forceplot_{}.html'.format(class_ind))

        if force_plot:
            if self.type_ == 'cls':
                shap.save_html(f, shap.force_plot(explainer.expected_value[class_ind],
                                                     shap_vals[class_ind], X_test_plot, show=False))
            elif self.type_ == 'reg':
                shap.save_html(f, shap.force_plot(explainer.expected_value, shap_vals,
                                                     X_test_plot, show=False))
        if interactions:
            return self.shap_interaction_vals


    def unravel_report(self, report):
        new_d = {}
        for key in report.keys():
            p1 = report[key]
            if key == 'accuracy':
                new_d[key] = report[key]
            try:
                _ = float(key)
                for key2 in p1.keys():
                    new_key_name = str(key) + ' ' + str(key2)
                    new_d[new_key_name] = report[key][key2]

            except:
                try:
                    for key2 in p1.keys():
                        new_key_name = str(key) + ' ' + str(key2)
                        new_d[new_key_name] = report[key][key2]
                except:
                    pass
        return new_d

    def combine_dicts(self, d_all, d_add):

        for key in d_add.keys():
            if key in d_all.keys():
                d_all[key] = np.concatenate((d_all[key], np.array([d_add[key]])))

            else:
                d_all[key] = np.asarray([d_add[key]])
        return d_all

    def find_categorial(self, df):
        likely_cat = {}
        for col in df.columns:
            values = np.unique(df[col].values)
            if ((np.round(values) - values).sum()) != 0:
                likely_cat[col] = False
            else:
                likely_cat[col] = 1. * df[col].nunique() / df[col].count() < 0.05  # or some other threshold
        return likely_cat

    def imblearn_rep_to_df(self, re):
        # from https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        # users: kindjacket and vlad calin buzea
        report_data = []
        lines = re.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[1]
            row['pre'] = float(row_data[2])
            row['rec'] = float(row_data[3])
            row['spe'] = float(row_data[4])
            row['f1'] = float(row_data[5])
            row['geo'] = float(row_data[6])
            row['iba'] = float(row_data[7])
            row['sup'] = float(row_data[8])
            report_data.append(row)
        return pd.DataFrame.from_dict(report_data)

    def save_kfold_summary(self, re):
        means = re.groupby(['class']).mean()
        ses = re.groupby(['class']).std()/np.sqrt(self.k)
        re.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                    '_k_fold_test_classification_report.csv'), index=False)
        means.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                    '_k_fold_test_classification_report_mean.csv'), index=False)
        ses.to_csv(os.path.join(self.output_dir, str(self.type_) + '_' + str(self.class_) +
                                  '_k_fold_test_classification_report_se.csv'), index=False)
