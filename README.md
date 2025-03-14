# Recipe-Rating-Prediction
Explore factors that could affect recipe rating, and predict ratings.

Authors: Fiona Zou, Ruofei Mao

## Overview


## Introduction
For the project, we will be using the a subset of recipes and ratings dataset from food.com. First, we will load in the two datasets and explore their features. 
The dataframe raw_recipes contains 83782 rows(each representing a unique recipe as indicated by the number of unique recipe IDs equals the number of total rows in this dataframe) and 12 columns (features) describing recipes. Specifically, the columns are:

1. name(object/string)- recipes' names stored as text data;
2. id(int)- numerical representation of recipe id;
3. minutes(int)- number of minutes to prepare the recipe;
4. contributor_id(int)- users' id who posted the recipe;
5. submitted(object/date) - date the recipe was submitted;
6. tags(object) - a list of strings representing tags for the recipe;
7. nutrition(object) - 7 nutrition information of the recipe, including #calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates in order of appearance;
8. n_steps(int) -number of steps in recipe;
9. steps(objects) - in-order text data for recipe steps;
10. description(object) -user descriptions of their recipe;
11. ingredients(object) -list of strings each representing a needed ingredient of recipe;
12. n_ingredients(int) -total number of recipe ingredients.


## Data Cleaning and Exploratory Data Analysis


## Assessment of Missingness


## Hypothesis Testing


## Framing a Prediction Problem

We will train a **classification model** to predict **average ratings of recipes**. The baseline model used a Decision Tree Classifier with a depth limit of 5 to predict recipe ratings as High, Medium, or Low.  The target variable, rating_category, was **ordinal and encoded as Low (0), Medium (1), and High (2)**. 

Since we want to predict the average rating of an recipe, we choose only features about the recipe's original qualities. Features we selected are the ones that are available at the “time of prediction”; that is, from the raw_recipe file. We choose average_rating as our response variable since it is a good indicator of the overall rating of a recipe. 

To evaluate model performance, overall accuracy and F1-score for each rating category were used as the primary metrics. Accuracy provides a general measure of how well the model performs across all predictions, while F1-score balances precision and recall, offering a better assessment for individual rating categories, especially when class distributions are imbalanced. Since the dataset is dominated by High ratings, relying solely on accuracy would be misleading, as the model could achieve high accuracy simply by predicting "High" most of the time. Therefore, the F1-score per class is essential to assess whether the model correctly identifies Medium and Low ratings, which are less frequent but still important.


## Baseline Model

For our baseline model, we are utilizing a decision tree classifier and split the data points into training and test sets. The model included **seven quantitative features**: calories, total fat percentage daily value (PDV), sugar PDV, sodium PDV, protein PDV, saturated fat PDV, and carbohydrates PDV. Since all features were numerical, no nominal variables required one-hot encoding. To ensure a more stable and accurate model, numerical features were standardized using StandardScaler() before training.

The baseline model achieved an overall accuracy of 87.64%, but a closer look at the F1-scores revealed major shortcomings. The model performed well on High ratings (F1 = 0.93) but performed poorly on Medium (F1 = 0.02) and Low (F1 = 0.00). This means that while the model correctly classified most High ratings, it almost completely failed to distinguish Medium and Low ratings, likely because of severe class imbalance. As a result, while the accuracy appears high, the model is not useful for predicting lower-rated recipes, which limits its practical value.


## Final Model


## Fairness Analysis
