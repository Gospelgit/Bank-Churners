'''
Note: It is best for each block of code to be run in Google Colab cells

'''


# Installing the libraries with the specified version.
!pip install scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==2.0.3 imbalanced-learn==0.10.1 xgboost==2.0.3 -q --user

#installing libraries incase previous step had issues 
import pandas as pd
import seaborn as sns
import numpy as np

# To split the data
from sklearn.model_selection import train_test_split

# To impute missing values
from sklearn.impute import SimpleImputer

# To build a Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier



# To undersample and oversample the data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To tune a model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# To get different performance metrics
import sklearn.metrics as metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# to build logstic regression model
from sklearn.linear_model import LogisticRegression

# to create k folds of data and get cross validation score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score




import warnings
warnings.filterwarnings('ignore')




#Data loading and overview 
from google.colab import files
df = files.upload()
df = pd.read_csv('BankChurners.csv')

df.head(10)

df.describe()

df.shape 

df.info() 

df['Atrrition_flag'].value_counts(1)

#Exploratory Data Analysis 
#function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(df, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    df: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=df, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=df, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        df[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        df[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# function to create labeled barplots


def labeled_barplot(df, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    df: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(df[feature])  # length of the column
    count = df[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=df,
        x=feature,
        palette="Paired",
        order=df[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

# function to plot stacked bar chart

def stacked_barplot(df, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    df: dataframe
    predictor: independent variable
    target: target variable
    """
    count = df[predictor].nunique()
    sorter = df[target].value_counts().index[-1]
    tab1 = pd.crosstab(df[predictor], df[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(df[predictor], df[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

### Function to plot distributions

def distribution_plot_wrt_target(df, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = df[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=df[df[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=df[df[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=df, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=df,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


#Univariate analysis 
sns.histplot(data=df, x ='Card_Category')

labeled_barplot(df,'Months_on_book')

sns.histplot(data=df, x ='Total_Ct_Chng_Q4_Q1')

sns.histplot(data=df,x='Total_Trans_Amt')
plt.show()



# creating histograms
df.hist(figsize=(14, 14))
plt.show()

sns.histplot(data=df,x='Education_Level')
plt.show()

labeled_barplot(df,'Education_Level')

df['Income_Category'].value_counts()

labeled_barplot(df,'Income_Category')

#Bivariate Analysis 

distribution_plot_wrt_target(df, 'Months_Inactive_12_mon', 'Attrition_Flag')

stacked_barplot(df, 'Months_Inactive_12_mon', 'Attrition_Flag')

distribution_plot_wrt_target(df, 'Total_Ct_Chng_Q4_Q1', 'Attrition_Flag')

sns.scatterplot(data = df, x = 'Total_Ct_Chng_Q4_Q1', y = 'Attrition_Flag')

distribution_plot_wrt_target(df, 'Months_on_book', 'Attrition_Flag')

stacked_barplot(df, 'Months_on_book', 'Attrition_Flag')

distribution_plot_wrt_target(df, 'Income_Category', 'Attrition_Flag')

distribution_plot_wrt_target(df, 'Avg_Utilization_Ratio', 'Attrition_Flag')

distribution_plot_wrt_target(df, 'Total_Relationship_Count', 'Attrition_Flag')


# Data preprocessing 

df1 = df.copy()

df1.isna.sum() 

cols = df1.select_dtypes(['object'])

for i in cols.columns:
    df1[i] = df1[i].astype('category')

# Clean the 'Income_Category' column to remove any leading or trailing spaces
df1['Income_Category'] = df1['Income_Category'].str.strip()

# Dictionary for replacing values
replaceStruct = {
    "Income_Category": {
        "Less than $40K": 1, "$40K - $60K": 2, "$60K - $80K": 3, "$80K - $120K": 4, "$120K +": 5, "abc": -1
    },
    "Attrition_Flag": {
        "Existing Customer": 0, "Attrited Customer": 1
    }
}

# Columns to apply one-hot encoding
oneHotCols = ['Card_Category', 'Marital_Status', 'Education_Level', 'Gender']

# Replace values in the DataFrame
df1.replace(replaceStruct, inplace=True)

# Apply one-hot encoding to the specified columns
df1 = pd.get_dummies(df1, columns=oneHotCols)

X = df1.drop(['Attrition_Flag', 'CLIENTNUM'],axis=1)
y = df1['Attrition_Flag']

# Splitting data into training, validation and test set:

# first we split data into 2 parts, say temporary and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

# then we split the temporary set into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=5, stratify=y_temp
)

print(X_train.shape, X_val.shape, X_test.shape)


# Missing Value Imputation 
imp_median = SimpleImputer(missing_values=np.nan, strategy="median")

# List of columns to impute
columns_to_impute = ["Education_Level_Doctorate", "Education_Level_Graduate", "Education_Level_Uneducated", "Education_Level_Post-Graduate", "Education_Level_High School", "Marital_Status_Married", "Marital_Status_Single"]

# Fit the imputer on train data and transform the train data
X_train[columns_to_impute] = imp_median.fit_transform(X_train[columns_to_impute])

# Transform the validation and test data using the imputer fit on train data
X_val[columns_to_impute] = imp_median.transform(X_val[columns_to_impute])
X_test[columns_to_impute] = imp_median.transform(X_test[columns_to_impute])


# Model Building 
# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1

        },
        index=[0],
    )

    return df_perf


## Function to create confusion matrix
def make_confusion_matrix(model, X, y):
    # Generate predictions
    y_pred = model.predict(X)

    # Compute the confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# Model building with original data 

models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(('AdaBoost', AdaBoostClassifier(random_state=1)))
models.append(('Gradient Boosting', GradientBoostingClassifier(random_state=1)))
models.append(('Decision Tree', DecisionTreeClassifier(random_state=1)))


print("\n" "Training Performance:" "\n")
for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))

# Model Building with Oversampled data 
# Synthetic Minority Over Sampling Technique
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)
models1 = []  # Empty list to store all the models

# Appending models into the list
models1.append(("Bagging", BaggingClassifier(random_state=1)))
models1.append(("Random forest", RandomForestClassifier(random_state=1)))
models1.append(('AdaBoost', AdaBoostClassifier(random_state=1)))
models1.append(('Gradient Boosting', GradientBoostingClassifier(random_state=1)))
models1.append(('Decision Tree', DecisionTreeClassifier(random_state=1)))


print("\n" "Training Performance:" "\n")
for name, model in models1:
    model.fit(X_train_over, y_train_over)
    scores = recall_score(y_train_over, model.predict(X_train_over))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in models1:
    model.fit(X_train_over, y_train_over)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))

# Model building with undersampled data 
# Random undersampler for under sampling the data
rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)

models2 = []  # Empty list to store all the models

# Appending models into the list
models2.append(("Bagging", BaggingClassifier(random_state=1)))
models2.append(("Random forest", RandomForestClassifier(random_state=1)))
models2.append(('AdaBoost', AdaBoostClassifier(random_state=1)))
models2.append(('Gradient Boosting', GradientBoostingClassifier(random_state=1)))
models2.append(('Decision Tree', DecisionTreeClassifier(random_state=1)))


print("\n" "Training Performance:" "\n")
for name, model in models2:
    model.fit(X_train_un, y_train_un)
    scores = recall_score(y_train_un, model.predict(X_train_un))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in models1:
    model.fit(X_train_un, y_train_un)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))


# Hyperparameter Tuning 

# Decision Tree with hypertuning on original data 

DT1 = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,6),
              'min_samples_leaf': [1, 4, 7],
              'max_leaf_nodes' : [10,15],
              'min_impurity_decrease': [0.0001,0.001] }

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=DT1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

# building model with best hyperparameters provided 
DT1_tuned = DecisionTreeClassifier(
    min_samples_leaf=7,
    min_impurity_decrease=0.0001,
    max_leaf_nodes=15,
    max_depth=5,
    random_state=1
)

# Fit the best algorithm to the data
DT1_tuned.fit(X_train, y_train)

#checking model performance 
dt1_perf = model_performance_classification_sklearn(DT1_tuned, X_train, y_train)

print(dt1_perf)

make_confusion_matrix(DT1_tuned,X_train, y_train)

#Hypertuned decision tree with oversampled data 
# defining model
DT2 = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,6),
              'min_samples_leaf': [1, 4, 7],
              'max_leaf_nodes' : [10,15],
              'min_impurity_decrease': [0.0001,0.001] }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=DT2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

DT2_tuned = DecisionTreeClassifier(
    min_samples_leaf=7,
    min_impurity_decrease=0.0001,
    max_leaf_nodes=15,
    max_depth=5,
    random_state=1
)

# Fit the best algorithm to the data
DT2_tuned.fit(X_train_over, y_train_over)

dt2_perf = model_performance_classification_sklearn(DT2_tuned, X_train_over, y_train_over)

print(dt2_perf)

make_confusion_matrix(DT2_tuned,X_train_over, y_train_over)


# Decision Tree with Hypertuning and undersampling
# defining model
DT3 = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,6),
              'min_samples_leaf': [1, 4, 7],
              'max_leaf_nodes' : [10,15],
              'min_impurity_decrease': [0.0001,0.001] }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=DT3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

DT3_tuned = DecisionTreeClassifier(
    min_samples_leaf=7,
    min_impurity_decrease=0.0001,
    max_leaf_nodes=15,
    max_depth=5,
    random_state=1
)

# Fit the best algorithm to the data
DT3_tuned.fit(X_train_un, y_train_un)

dt3_perf = model_performance_classification_sklearn(DT3_tuned, X_train_un, y_train_un)


print(dt3_perf)

make_confusion_matrix(DT3_tuned,X_train_un, y_train_un)

# Gradient Boosting Hypertuning

#with original data 

GB1 = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "init": [AdaBoostClassifier(random_state=1),DecisionTreeClassifier(random_state=1)],
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "subsample":[0.7,0.9],
    "max_features":[0.5,0.7,1],
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=GB1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


GB1_tuned = GradientBoostingClassifier(
    subsample=0.7,
    n_estimators=100,
    max_features=0.7,
    learning_rate=0.05,
    init=DecisionTreeClassifier(random_state=1)
)

# Fit the best algorithm to the data
GB1_tuned.fit(X_train, y_train)

gb1_perf = model_performance_classification_sklearn(GB1_tuned, X_train, y_train)

print(gb1_perf)

make_confusion_matrix(GB1_tuned,X_train, y_train)


# Gradient Boosting for Oversampled Data
# defining model
GB2 = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "init": [AdaBoostClassifier(random_state=1),DecisionTreeClassifier(random_state=1)],
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "subsample":[0.7,0.9],
    "max_features":[0.5,0.7,1],
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=GB2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)
print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

GB2_tuned = GradientBoostingClassifier(
    subsample=0.7,
    n_estimators=100,
    max_features=0.7,
    learning_rate=0.05,
    init=DecisionTreeClassifier(random_state=1)
)

# Fit the best algorithm to the data
GB2_tuned.fit(X_train_over, y_train_over)

gb2_perf = model_performance_classification_sklearn(GB2_tuned, X_train_over, y_train_over)
print(gb2_perf)

make_confusion_matrix(GB2_tuned,X_train_over, y_train_over)

# defining model
GB3 = GradientBoostingClassifier(random_state=1)

param_grid = {
    "init": [AdaBoostClassifier(random_state=1),DecisionTreeClassifier(random_state=1)],
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "subsample":[0.7,0.9],
    "max_features":[0.5,0.7,1],
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=GB3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

GB3_tuned = GradientBoostingClassifier(
    subsample=0.7,
    n_estimators=100,
    max_features=0.7,
    learning_rate=0.05,
    init=DecisionTreeClassifier(random_state=1)
)

# Fit the best algorithm to the data
GB3_tuned.fit(X_train_un, y_train_un)


gb3_perf = model_performance_classification_sklearn(GB3_tuned, X_train_un, y_train_un)
print(gb3_perf)


make_confusion_matrix(GB3_tuned,X_train_un, y_train_un)

# AdaBoost Classifier

#AdaBoost Hypertuned with original data 

AB1 = AdaBoostClassifier(random_state=1)

param_grid = {
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "base_estimator": [
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=AB1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))
AB1_tuned = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=1),
    n_estimators=100,
    learning_rate=0.1
)

# Fit the best algorithm to the data
AB1_tuned.fit(X_train, y_train)


DecisionTreeClassifier
ab1_perf = model_performance_classification_sklearn(AB1_tuned, X_train, y_train)

print(ab1_perf)

make_confusion_matrix(AB1_tuned,X_train, y_train)

AdaBoost with oversampled data
# defining model
AB2 = AdaBoostClassifier(random_state=1)

param_grid = {
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "base_estimator": [
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=AB2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

AB2_tuned = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=1),
    n_estimators=100,
    learning_rate=0.1
)

# Fit the best algorithm to the data
AB2_tuned.fit(X_train_over, y_train_over)


# DecisionTreeClassifier
ab2_perf = model_performance_classification_sklearn(AB2_tuned, X_train_over, y_train_over)

print(ab2_perf)

make_confusion_matrix(GB2_tuned,X_train_over, y_train_over)

# AdaBoost for Undersampled
# defining model
AB3 = AdaBoostClassifier(random_state=1)

param_grid = {
    "n_estimators": np.arange(50,110,25),
    "learning_rate": [0.01,0.1,0.05],
    "base_estimator": [
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=AB3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

AB3_tuned = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3, random_state=1),
    n_estimators=100,
    learning_rate=0.1
)

# Fit the best algorithm to the data
AB3_tuned.fit(X_train_un, y_train_un)


AB3_perf = model_performance_classification_sklearn(AB3_tuned, X_train_un, y_train_un)


print(AB3_perf)

make_confusion_matrix(AB3_tuned,X_train_un, y_train_un)

#Bagging Classifier
#Bagging Classifier with original data
BC1 = BaggingClassifier(random_state=1)

param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70],
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=BC1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

BC1_tuned = BaggingClassifier(
    max_samples=0.8,
    n_estimators=70,
    max_features=0.7
)

# Fit the best algorithm to the data
BC1_tuned.fit(X_train, y_train)


bc1_perf = model_performance_classification_sklearn(BC1_tuned, X_train, y_train)

print(bc1_perf)

make_confusion_matrix(BC1_tuned,X_train, y_train)

#Bagging Classifier with Oversampled
# defining model
BC2 = BaggingClassifier(random_state=1)

param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70],
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=BC2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

BC2_tuned = BaggingClassifier(
    max_features = 0.7,
    n_estimators = 70,
    max_samples = 0.8
)


# Fit the best algorithm to the data
BC2_tuned.fit(X_train_un, y_train_un)


bc2_perf = model_performance_classification_sklearn(BC2_tuned, X_train_over, y_train_over)


print(bc2_perf)

make_confusion_matrix(BC2_tuned, X_train_over, y_train_over)

#Bagging with undersampling
# defining model
BC3 = BaggingClassifier(random_state=1)


param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70],
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=BC3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

BC3_tuned = BaggingClassifier(
    max_features=0.7,
    n_estimators=75,
    max_samples=0.8
)

# Fit the best algorithm to the data
BC3_tuned.fit(X_train_un, y_train_un)


bc3_perf = model_performance_classification_sklearn(BC3_tuned, X_train_un, y_train_un)

print(bc3_perf)

make_confusion_matrix(BC3_tuned,X_train_un, y_train_un)

#Random Forest Classifier
#Hypertuned Random forest with original data
RF1 = RandomForestClassifier(random_state=1)

param_grid = {
    "n_estimators": [50,110,25],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=RF1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

RF1_tuned = RandomForestClassifier(
    max_samples=0.6,
    n_estimators=110,
    max_features='sqrt',
    min_samples_leaf=1
)

# Fit the best algorithm to the data
RF1_tuned.fit(X_train, y_train)


rf1_perf = model_performance_classification_sklearn(RF1_tuned, X_train, y_train)

print(rf1_perf)

make_confusion_matrix(RF1_tuned,X_train, y_train)

#Building Random Forest with oversampled data
# defining model
RF2 = RandomForestClassifier(random_state=1)

param_grid = {
    "n_estimators": [50,110,25],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)
}

scorer = make_scorer(recall_score, average='weighted')

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=RF2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

RF2_tuned = RandomForestClassifier(
    max_features = 'sqrt',
    n_estimators = 110,
    max_samples = 0.6,
    min_samples_leaf=1
)


# Fit the best algorithm to the data
RF2_tuned.fit(X_train_over, y_train_over)


rf2_perf = model_performance_classification_sklearn(RF2_tuned, X_train_over, y_train_over)


print(rf2_perf)

make_confusion_matrix(RF2_tuned, X_train_over, y_train_over)


#Random Forest with undersampling
# defining model
RF3 = RandomForestClassifier(random_state=1)

param_grid = {
    "n_estimators": [50,110,25],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=RF3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

RF3_tuned = RandomForestClassifier(
    max_features='sqrt',
    n_estimators=110,
    max_samples=0.5,
    min_samples_leaf=2
)

# Fit the best algorithm to the data
RF3_tuned.fit(X_train_un, y_train_un)

rf3_perf = model_performance_classification_sklearn(RF3_tuned, X_train_un, y_train_un)


print(rf3_perf)


#XGBoost Model
#XGBoost with original data
XG1 = XGBClassifier(random_state=1)

param_grid={'n_estimators':np.arange(50,110,25),
            'scale_pos_weight':[1,2,5],
            'learning_rate':[0.01,0.1,0.05],
            'gamma':[1,3],
            'subsample':[0.7,0.9]
}

scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=XG1, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

XG1_tuned = XGBClassifier(
    learning_rate=0.05,
    n_estimators=110,
    gamma=1,
    subsample=0.7,
    scale_pos_weight=1
)
# Fit the best algorithm to the data
XG1_tuned.fit(X_train, y_train)

xg1_perf = model_performance_classification_sklearn(XG1_tuned, X_train, y_train)

print(xg1_perf)

make_confusion_matrix(XG1_tuned,X_train, y_train)

#XGBoost with Oversampled data
# defining model
XG2 = XGBClassifier(random_state=1)

param_grid={'n_estimators':np.arange(50,110,25),
            'scale_pos_weight':[1,2,5],
            'learning_rate':[0.01,0.1,0.05],
            'gamma':[1,3],
            'subsample':[0.7,0.9]

}
scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=XG2, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

XG2_tuned = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    gamma=3,
    subsample=0.7,
    scale_pos_weight=5
)

# Fit the best algorithm to the data
XG2_tuned.fit(X_train_over, y_train_over)

xg2_perf = model_performance_classification_sklearn(XG2_tuned, X_train_over, y_train_over)


print(xg2_perf)

make_confusion_matrix(XG2_tuned, X_train_over, y_train_over)

#XGboost with undersampling data
# defining model

XG3 = XGBClassifier(random_state=1)
param_grid={'n_estimators':np.arange(50,110,25),
            'scale_pos_weight':[1,2,5],
            'learning_rate':[0.01,0.1,0.05],
            'gamma':[1,3],
            'subsample':[0.7,0.9]
}
scorer = make_scorer(recall_score, average='weighted')

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=XG3, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))

XG3_tuned =XGBClassifier(
    learning_rate=0.05,
    n_estimators=100,
    gamma=1,
    subsample=0.7,
    scale_pos_weight=1
)

# Fit the best algorithm to the data
XG3_tuned.fit(X_train_un, y_train_un)

xg3_perf = model_performance_classification_sklearn(XG3_tuned, X_train_un, y_train_un)


print(xg3_perf)

make_confusion_matrix(XG3_tuned,X_train_un, y_train_un)

# Model Comparison and Final Model Selection
# Checking for recall scores of Models hypertuned on Original data
HyperOriginals = [] # Empty list to store all the models

# Appending models into the list
HyperOriginals.append(('Decision Tree1', DT1_tuned))
HyperOriginals.append(('Gradient Boosting1', GB1_tuned))
HyperOriginals.append(('AdaBoost1', AB1_tuned))
HyperOriginals.append(('Bagging Classifier1', BC1_tuned))
HyperOriginals.append(('Random Forest1', RF1_tuned))
HyperOriginals.append(('XGBoost1', XG1_tuned))

print("\n" "Training Performance:" "\n")
for name, model in HyperOriginals:
    model.fit(X_train, y_train)
    scores = recall_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in HyperOriginals:
    model.fit(X_train, y_train)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))

# Checking for recall scores for models hypertuned on oversampled data
HyperOversampled = [] # Empty list to store all the models

# Appending models into the list
HyperOversampled.append(('Decision Tree 2', DT2_tuned))
HyperOversampled.append(('Gradient Boosting2', GB2_tuned))
HyperOversampled.append(('AdaBoosting2', AB2_tuned))
HyperOversampled.append(('Bagging Classifier2', BC2_tuned))
HyperOversampled.append(('RandomForest2', RF2_tuned))
HyperOversampled.append(('XGBoost2', XG2_tuned))

print("\n" "Training Performance:" "\n")
for name, model in HyperOversampled:
    model.fit(X_train, y_train)
    scores = recall_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in HyperOversampled:
    model.fit(X_train, y_train)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))


#Checking for recall scores of Hypertuned models on undersampled data
HyperUndersampled = [] # Empty list to store all the models

# Appending models into the list
HyperUndersampled.append(('Decision Tree 3', DT3_tuned))
HyperUndersampled.append(('Gradient Boosting3', GB3_tuned))
HyperUndersampled.append(('AdaBoosting3', AB3_tuned))
HyperUndersampled.append(('Bagging Classifier3', BC3_tuned))
HyperUndersampled.append(('RandomForest3', RF3_tuned))
HyperUndersampled.append(('XGBoost3', XG3_tuned))

print("\n" "Training Performance:" "\n")
for name, model in HyperUndersampled:
    model.fit(X_train, y_train)
    scores = recall_score(y_train, model.predict(X_train))
    print("{}: {}".format(name, scores))

print("\n" "Validation Performance:" "\n")

for name, model in HyperUndersampled:
    model.fit(X_train, y_train)
    scores_val = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores_val))

Creating a dataframe for all models
import pandas as pd

# Define the data for each set with clarified suffixes
data = {
    'Model': [
        'Decision Tree (Original)', 'Gradient Boosting (Original)', 'AdaBoost (Original)', 'Bagging Classifier (Original)', 'Random Forest (Original)', 'XGBoost (Original)',
        'Decision Tree (Oversampled)', 'Gradient Boosting (Oversampled)', 'AdaBoost (Oversampled)', 'Bagging Classifier (Oversampled)', 'Random Forest (Oversampled)', 'XGBoost (Oversampled)',
        'Decision Tree (Undersampled)', 'Gradient Boosting (Undersampled)', 'AdaBoost (Undersampled)', 'Bagging Classifier (Undersampled)', 'Random Forest (Undersampled)', 'XGBoost (Undersampled)'
    ],
    'Training Performance': [
        0.8069164265129684, 1.0, 0.9298751200768491, 1.0, 0.9807877041306436, 0.9500480307396734,
        0.8069164265129684, 1.0, 0.9298751200768491, 0.9980787704130644, 0.9827089337175793, 1.0,
        0.8069164265129684, 1.0, 0.9298751200768491, 0.9990393852065321, 0.9077809798270894, 0.9442843419788665
    ],
    'Validation Performance': [
        0.8237547892720306, 0.8045977011494253, 0.8773946360153256, 0.8275862068965517, 0.7509578544061303, 0.8850574712643678,
        0.8237547892720306, 0.8045977011494253, 0.8773946360153256, 0.842911877394636, 0.7471264367816092, 0.946360153256705,
        0.8237547892720306, 0.8045977011494253, 0.8773946360153256, 0.8275862068965517, 0.7394636015325671, 0.8812260536398467
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set display options to avoid wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print DataFrame
print(df)
                               
# Test set final performance
# Checking recall score on test set
print("AB Oversampled on test set")
print(recall_score(y_test, AB2_tuned.predict(X_test)))
print("")


# Checking recall score on test set
print("AB Undersampled on test set")
print(recall_score(y_test, AB3_tuned.predict(X_test)))
print("")


gb_model = GradientBoostingClassifier(random_state=1)

# Fit the model on the training data
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Calculate and print the recall score
print("GB with no hypertuning on test set")
print(recall_score(y_test, y_pred))
print("")


importances = AB2_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
