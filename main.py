import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import shap

#Read in file
df = pd.read_csv('DQN1 Dataset.csv')

# Convert date time epoch to time of day
df['datetime'] = pd.to_datetime(df['datetimeEpoch'], unit='s', utc=True)

#Extract the hour (0-23)
df['hour'] = df['datetime'].dt.hour

#Define bins and corresponding labels
bins = [0, 5, 12, 17, 21, 24]
labels = ['night', 'morning', 'day', 'evening', 'night']

# Use the Pandas factorize method to map the categorical values to integers
df['isWeekend'] = pd.factorize(df['isWeekend'])[0]

#Convert hour to time of day
df['time_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels, include_lowest=True, right=False, ordered=False)

#Label encode time of day
df['time_of_day'] = LabelEncoder().fit_transform(df['time_of_day'])

#Get day of month
df['date_time'] = pd.to_datetime(df['datetimeEpoch'], unit='s')

# 2. Extract the day of the month as a new feature
df['day_of_month'] = df['date_time'].dt.day

#print(df.var(numeric_only=True))

df.drop(columns=['datetimeEpoch'], inplace=True)

#Create heat map
'''plt.figure(figsize=(40, 40))
sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True, annot_kws={"fontsize": 7})
plt.show()'''

#Create a new feature that is the difference between feelslikemax and feelslikemin.
df['subjectiveTempDiff'] = df['feelslikemax'] - df['feelslikemin']

#Create a new feature that is the difference between sunriseEpoch and sunsetEpoch.
df['durationOfSunlight'] = abs(df['sunriseEpoch'] - df['sunsetEpoch'])

#Features that are to be removed
df = df.drop(columns=['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'day_of_month', 'date_time',
                      'sunriseEpoch', 'sunsetEpoch', 'windgust', 'solarenergy','month','dayOfWeek', 'datetime', 'hour'])

#Heat map showing changes
'''plt.figure(figsize=(40, 40))
sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True, annot_kws={"fontsize": 7})
plt.show()'''

#Drop tempRange
df.drop(columns=['tempRange'], inplace=True)

#Seperate independent and dependent variables
X = df.iloc[:, df.columns != 'healthRiskScore']
y = df['healthRiskScore']

#Compute VIF
# 2. Add an intercept/constant column for the regression underlying VIF
X = add_constant(X)

#Drop features with high vif values
X.drop(columns=['solarradiation', 'heatIndex', 'humidity'], inplace=True)

# 3. Compute VIF for each feature
vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# 4. View results (usually you can drop the 'const' row from the final output)
print(vif_df[vif_df["Variable"] != "const"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Gradient Boosting Regression
gbr =  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# 4. Define the hyperparameter search grid
param_distributions = {
    'n_estimators': range(10, 101),
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': range(1, 50),
    'min_samples_split': range(2, 22),
    'min_samples_leaf': range(1, 11),
    'max_features': ['sqrt', 'log2'],
    }

# 5. Set up Grid Search with 5-fold cross-validation
random_search_gbr = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='r2', # Evaluation metric
    verbose=1,
    random_state=42,
    n_jobs=-1 # Use all available CPU cores
)

#Fit the model to the training data
random_search_gbr.fit(X_train, y_train)

#Results of cross validation
best_index = random_search_gbr.best_index_

#print(f"Best Parameters: {random_search_gbr.best_params_}\n")

print("R² values for each cross-validation fold:")
for fold in range(random_search_gbr.cv):
    fold_score = random_search_gbr.cv_results_[f"split{fold}_test_score"][best_index]
    print(f"  Fold {fold + 1}: {fold_score:.4f}")

print(f"\nMean R² Score: {random_search_gbr.best_score_:.4f}")
#print("best params", random_search_gbr.best_params_)

#Make predictions on test data using best parameters
y_pred = random_search_gbr.predict(X_test)

#Calculate R-Squared
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

#Calculate rmse
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

#plot shap values (feature importances)

explainer = shap.TreeExplainer(random_search_gbr.best_estimator_)
shap_values = explainer.shap_values(X_test)

#Plot overall feature impact
shap.summary_plot(shap_values, X_test)

#Force plot
#shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0, :], matplotlib=True)

