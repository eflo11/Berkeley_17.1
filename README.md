# Berkeley_17.1
## Campaign Accuracy
With in this Jupyter Notebook you will find the analysis of banking campaigns that allows us to make more data driven decisions on how to better predict who will sign up to enroll in long term deposits with good interest rates. The full definition of the work can be found [here](./CRISP-DM-BANK.pdf).

## Analysis
To first make predictions we first need to understand the data. This is done through various techniques. The first part was to check for incomplete data:
```python
for column in df.columns:
    null_count = df[column].isnull().sum()
    if null_count > 0:
        print(f"Column '{column}' has {null_count} null values.")
```
This will give me the results of any columns that have null values. Once you determine which columns may be missing data you have to determine whether or not you get rid of the data, populate with averages.

For this dataset it was already cleaned before it was received by us so we don't have to make up any values for the nulls.

## Preparation
The next step is to go through the process of encoding and scaling the data. This will include splitting away your target to the other columns and then work on converting categorical and numerical columns.
```python
def encode(df):
    # Encode the target column 'y'
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['y'])
    X = df.drop(columns=['y'], axis=1)

    # Encode all categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['number']).columns

    # Encode categorical columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Encode numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    return X, y
```

## Modeling
The next phase it to start working on trying out the different modeling options. This includes first creating a DummyClassifier so that it gives us a good default threshold we can create a base off of.

Once we have created the base, it gave us a decent 88.76% accuracy. The next phase is to run the LogisticRegression, DT, KNN, and SVC. Each of those models did better than the threshold.

![Model Performance Comparison](./images/initial_model_performance_comparison.png)

With these results it helps show us the both LogisticRegression and SVC were the top two just separated by a few hundredths of percent.

## Re-analysis
The next phase was to see if we could get anything better than 91%. To do this I went through a lot of the categorical attributes to see if we could find any that have the same percentages for yes/no to where if we removed them it would speed up the modeling process and still not affect the result.

For this I found a couple that I could combine/get rid of. The first was grouping some of the marital values. I could combine like values to reduce the amount of columns.
```
y                no        yes
marital                       
divorced  89.679098  10.320902
married   89.842747  10.157253
single    85.995851  14.004149
unknown   85.000000  15.000000
```

Next we ended up getting rid of day of the week. They were close together where it wouldn't make a huge impact to the results.
```
y                   no        yes
day_of_week                      
fri          89.191261  10.808739
mon          90.051680   9.948320
thu          87.881248  12.118752
tue          88.220025  11.779975
wed          88.332924  11.667076
```

## Outcome
After doing another round of models we were able to determine that the DT through different params using GridSearch and got it to a 91.4% accuracy which was a 3% increase vs initial testing.

However when looking at the overall graph that includes the accuracy, precision, and recall there is one that has a higher overall value when considering those three tiers and that is DT.

![Result](./images/results.png)

With the accuracy so close between all of them it makes sense to then give equal weight to precision, and recall to make sure that you are receiving less false positives and false negatives.