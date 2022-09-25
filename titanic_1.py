### this code is my attempt to predict the passengers that died onboard the Titanic, using Kaggle's titanic competition data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from scipy.stats import pearsonr
from sklearn import linear_model

#creating dataframe from files

df_gender = pd.read_csv('gender_submission.csv')

df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')


display(df_train)

#creating empty lists for correlation names and their corresponding values

corr_coeff_name = []
corr_coeff_values = []

#calculating pearson coefficient for passenger class and survival

survived = df_train['Survived']
p_class = df_train['Pclass']
p_class_coeff, _ = pearsonr(survived, p_class)
print('The pearson correlation coefficient for passenger class is %.5f:' % p_class_coeff)
corr_coeff_name.append('passenger_class_coeff')
corr_coeff_values.append(p_class_coeff)

# creating a distribution plot for passenger class and survival rate
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_train['Pclass'][df_train.Survived == 1], color='darkturquoise', shade=True)
sns.kdeplot(df_train['Pclass'][df_train.Survived == 0], color='lightcoral', shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Distribution Plot of Passenger Class for Surviving and Deceased Passengers')
plt.show()
plt.close()

### we can see from this distribution plot that third class passengers made up the vast majority of the total deaths,
### while first class had the highest number of surviving passengers, but a much higher percentage of passengers that survived,
### compared to the other two classes.

# creating a vertical bar graph of survival between passenger classes, and segmented by sex
plt.figure(figsize=(15,8))
ax = sns.barplot(x=df_train['Pclass'], y=df_train['Survived'], hue=df_train['Sex'], data=df_train)
plt.title('Bar Graph of Passenger Class and Survival with Sex Differentiation')
plt.show()
plt.close()

### as we can see from this bar plot, women had a much higher survival rate than men, and first class passengers
### also had a much higher survival rate than third class. Nearly all of the first class female passengers survived.

#calculating the pearson coefficient for passenger fare and survival
fare = df_train['Fare']
fare_coeff, _ = pearsonr(survived, fare)
print('The pearson correlation coefficient for fare price is %.5f:' % fare_coeff)
corr_coeff_name.append('fare_coeff')
corr_coeff_values.append(fare_coeff)

#calculating pearson coefficient for parch column (number of parents/children on board)
parch = df_train['Parch']
parch_coeff, _ = pearsonr(survived, parch)
print('The pearson correlation coefficient for parch is %.5f:' % parch_coeff)
corr_coeff_name.append('parch_coeff')
corr_coeff_values.append(parch_coeff)

#calculating pearson correlation coefficient for SibSp (# of siblings/spouses on board)
sibSp = df_train['SibSp']
sibSp_coeff, _ = pearsonr(survived, sibSp)
print('The pearson correlation coefficient for SibSp is %.5f:' % sibSp_coeff)
corr_coeff_name.append('SibSp_coeff')
corr_coeff_values.append(sibSp_coeff)

#calculating pearson coefficient for Age

df_age = df_train[['Survived', 'Age']]
df_age = df_age.dropna()
age = df_age['Age']
age_survived = df_age['Survived']
age_coeff, _ = pearsonr(age_survived, age)
print('The pearson correlation coefficient for Age is %.5f:' % age_coeff)
corr_coeff_name.append('age_coeff')
corr_coeff_values.append(age_coeff)

#creating new dataframe with columns survived and embarked
                                                                                                                                     
df_embarked_survived = df_train[['Survived', 'Embarked']]
print(df_embarked_survived.value_counts())

# creating dataframe with columns survived and sex

df_sex = df_train[['Survived', 'Sex']]
print(df_sex.value_counts())

ratio_men_survived = (109 / (468+109)) * 100
ratio_women_survived = (233 / (233 + 81)) * 100
print('The percentage of men that survived was %.3f,' % ratio_men_survived, 'And the percentage of women that survived was %.3f.' % ratio_women_survived)

#creating numerical values for each sex so we can calculate the correlation coefficient between sex and survival

sex_values = []
for i in df_sex['Sex']:
    if i == 'male':
        sex_values.append(1)
    else:
        sex_values.append(2)


df_sex['SexValue'] = sex_values
df_sex.drop(['Sex'], axis=1)
        
sex_coeff, _ = pearsonr(survived, df_sex['SexValue'])
print('The pearson correlation for Sex is %.5f' % sex_coeff)
corr_coeff_name.append('sex_coeff')
corr_coeff_values.append(sex_coeff)

plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_train['Age'][df_train.Survived == 1], color='darkturquoise', shade=True)
sns.kdeplot(df_train['Age'][df_train.Survived == 0], color='lightcoral', shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Distribution Plot of Age for Surviving and Deceased Passengers')
plt.show()
plt.close()

### as we can see from the pearson coefficient for age and the distribution plot, age had almost no correlation with survival

#calculating correlation between fare price and passenger class
fare_class_coeff, _ = pearsonr(p_class, fare)
print('The pearson coefficient between passenger class and fare price is %.5f' % fare_class_coeff)
corr_coeff_name.append('fare_class_coeff')
corr_coeff_values.append(fare_class_coeff)

# creating a dictionary to store various correlation values

coeff_dict = dict(zip(corr_coeff_name, corr_coeff_values))
print(coeff_dict)

# creating multiple linear regression models to predict passenger deaths

X1 = df_train['Fare']
X2 = df_sex['SexValue']
X3 = df_train['Pclass']
X_train = pd.concat([X1, X2, X3], axis=1, join='inner')

#creating X_test dataframe to use for model prediction

df_sex_test = df_test[['Sex']]
sex_values_test = []
for i in df_sex_test['Sex']:
    if i == 'male':
        sex_values_test.append(1)
    else:
        sex_values_test.append(2)


df_sex_test['SexValue'] = sex_values_test
df_sex_test.drop(['Sex'], axis=1)

avg_fare = df_test['Fare'].mean()

X_1 = df_test['Fare']
X_2 = df_sex_test['SexValue']
X_3 = df_test['Pclass']
X_test = pd.concat([X_1, X_2, X_3], axis=1, join='inner')

#dropping Nan values from X_test dataframe (one single row)

X_test = X_test.fillna(avg_fare)

Y = df_train['Survived']

reg1 = linear_model.LinearRegression()
model1 = reg1.fit(X_train, Y)
Y_hat_lin = reg1.predict(X_test)

print('Linear model intercept is:', reg1.intercept_)
print('Linear model coefficients are:', reg1.coef_)


# creating logistic regression model for the data

reg2 = linear_model.LogisticRegression()
model2 = reg2.fit(X_train, Y)
Y_hat_log = reg2.predict(X_test)

print('Logistic model intercept is:', reg2.intercept_)
print('Logistic model coefficients are:', reg2.coef_)

X_test['Survived'] = Y_hat_log
df_test['Survived'] = Y_hat_log

df_test_prediction = df_test[['PassengerId', 'Survived']]

display(df_test_prediction)

df_test_prediction.to_csv('LogisticRegression_Titanic_MJS.csv', index=False)







