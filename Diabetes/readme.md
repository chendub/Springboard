## Predicting Diabetes with the Pima Indians Disbetes Dataset

### Introduction

The Pima people are a group of Native Americans inhabiting an area in central and southern Arizona, as well as northwestern Mexico. The Pima people in Arizona are observed to have a very high prevalence of Type II Diabetes while the genetically identical Pima people living in Mexico do not. Therefore it is hypothesized that it is a combination of genetic and environmental factors that causes the high incidence of diabetes in the Pima people in Arizona. This also made them the subject of many studies on Type II Diabetes. The Pima Indians Diabetes dataset originated from one of such studies. The dataset consists of eight different diagnostic measurements and the outcome of whether or not the individuals have diabetes from a population of Pima Native American women of at least 21 years of age . The objective of this project is to build a machine learning model that predicts if individuals in this population have diabetes outcomes based on their diagnostic measurements. 

### Data Wrangling

The dataset is in csv format and has 9 columns. There are 8 independent variables: ‘Pregnancies’, ‘Glucose’, ‘BloodPressure’, ‘SkinThickness’, ‘Insulin’, ‘BMI’, ‘DiabetesPedigreeFunction’ and ‘Age’, as well as one binary ‘Outcome’ variable, where ‘1’ denotes that the individual was diagnosed with diabetes and ‘0’ means no diabetes. The dataset has 768 rows with no duplicate entries. All of the data are numerical non-nulls but missing values could be in the form of ‘0’s’. 

The dataset is fairly clean to begin with and the only task for data wrangling is to replace missing values with NaN values. There are no ‘0’ values in ‘DiabetesPedigreeFunction’ and ‘Age’. Out of the remaining  6 features, it only makes sense to have 0 ‘Pregnancies’ and therefore the ‘0’ in the rest of the other columns are assumed to be missing values and were replaced by NaN values. Of note, there are 374 missing ‘Insulin’ values and 227 missing ‘SkinThickness’ values, which means there are 48.7% and 29.6% missing values in the respective columns. 

### Exploratory Data Analysis

Boxplots (Figure 1.) were generated for each of the features grouped by ‘Outcome’.  At a glance, in all 8 features, the average values in individuals with diabetes are slightly higher than those without. However, only in the feature ‘Glucose’ do we see a clear and significant separation between the ranges of the two different ‘Outcome’ groups. Therefore ‘Glucose’ should be a feature of high importance in our model.

![Boxplots of values in each feature grouped by 'Outcome'.](https://github.com/chendub/Springboard/blob/main/Diabetes/images/box.png)
Figure 1. Boxplots of values in each feature grouped by ‘Outcome’.

### Pre-Processing

The values in each feature were plotted as histograms (Figure 2). The distribution for ‘BloodPressure’ appears to be close to Normal while the distributions of the other 7 features display skewness of varying degree. In order to minimize the effect of outliers, for imputation of missing data, medians were used for all the features except for ‘BloodPressure’ due to its fairly Normal distribution. 

![Distribution of values in each feature before imputation of missing values.](https://github.com/chendub/Springboard/blob/main/Diabetes/images/histo.png)
Figure 2. Distribution of values in each feature before imputation of missing values.

Data was split into a 75% train set and 25% test set. The train and test sets were then scale transformed separately using the standard deviation (StandardScaler from sklearn). 

### Modeling

A simple Logistic Regression model was fitted initially without any hyperparameter tuning. The model is then evaluated using the accuracy score.  The training accuracy for this model is 0.78 while the testing accuracy is 0.73. We then tried to improve the model using GridSearchCV. The resulting best estimator has the parameters of ‘C’ = 10.0, ‘penalty’ = ‘l1’ and yields a training accuracy of 0.78 and testing accuracy of 0.75.

A simple Decision Tree model gives a training accuracy of 0.85 and testing accuracy of 0.71. We then used the GridSearchCV method to fit a Random Forest model. The best parameters were determined to have the following best parameters:  {'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 16, 'n_estimators': 300}. Using this model, we got a training accuracy of 0.88, and testing accuracy of 0.73.

A Support Vector Machine model was also used. Without any hyperparameter tuning, it gives a training accuracy of 0.85 and testing accuracy of 0.71. After hyperparameter tuning, the training accuracy is 0.77 and testing accuracy is 0.75.

Additionally, we looked at feature importances based on the Random Forest model (figure 3.) and refitted models with the 2 least important features (‘BloodPressure’ and ‘SkinThickness’) dropped. Best estimators for Logistic Regression, Random Forest and Support Vector Machine after GridSearchCV were refitted and we did see improvement in test metrics in all of three models. 

![Feature importances based on the Random Forest model.](https://github.com/chendub/Springboard/blob/main/Diabetes/images/featureimpotances.png)

Figure 3. Feature importances based on the Random Forest model.

The objective of the model should be to accurately predict diabetes outcome, while minimizing false positive and false negative predictions. Therefore the F-1 score for all of the models were also calculated.  The accuracy scores and F-1 scores are organized into the following tables:

Table 1: Comparing accuracy scores and F-1 scores for 3 models with no hyperparameter tuning:
|   |Logistic Regresson   |Decision Tree   |Support Vector Machine   |
|---|---|---|---|
|Traingin Accuracy  |0.78   |0.85   |0.85   |
|Testing Accuracy   |0.73   |0.71   |0.71   |
|F-1 score   |0.69   |0.66   |0.65   |

Table 2: Comparing accuracy scores and F-1 scores for 3 models with hyperparameter tuning using GridSearchCV:
|   |Logistic Regresson   |Random Forest   |Support Vector Machine   |
|---|---|---|---|
|Traingin Accuracy  |0.78   |0.88   |0.77   |
|Testing Accuracy   |0.75   |0.73   |0.75   |
|F-1 score   |0.70   |0.69   |0.70   |

Table 3: Comparing accuracy scores and F-1 scores for 3 models after GridSearchCV and dropping 2 features with least importance as determined by the Random Forest model:
|   |Logistic Regresson   |Random Forest   |Support Vector Machine   |
|---|---|---|---|
|Traingin Accuracy  |0.77   |0.88   |0.77   |
|Testing Accuracy   |0.78   |0.79   |0.77   |
|F-1 score   |0.75   |0.76   |0.73   |

Dropping the features ‘BloodPressure’ and ‘SkinThickness’ improves the test metrics in all 3 models. Although the Random Forest model has the highest accuracy score and F-1 score, its gap between training and testing accuracy indicates that it is potentially over-fitting. The Logistic Regression model has nearly as good of testing accuracy score and F-1 as Random Forest, and it might predict more accurately with new data.

### Conclusion

Using the 6 features ‘Glucose’, ‘BMI’, ‘Age’, ‘DiabetesPedigreeFunction’, ‘Pregnancies’ and ‘Insulin’ and using the parameters of {C=10.0, penalty=‘l1’, solver=‘liblinear’}, the Logistic Regression model can achieve an accuracy score of 0.78 and F-1score of 0.75 while not overfitting to the training data.  
