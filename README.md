# Recipe-Rating-Prediction
Explore factors that could affect recipe rating, and predict ratings.

Authors: Fiona Zou, Ruofei Mao

## Overview


## Introduction


## Data Cleaning and Exploratory Data Analysis


## Assessment of Missingness


## Hypothesis Testing


## Framing a Prediction Problem

We will train a **classification model** to predict **average ratings of recipes**. The baseline model used a Decision Tree Classifier with a depth limit of 5 to predict recipe ratings as High, Medium, or Low.  The target variable, rating_category, was ordinal and encoded as Low (0), Medium (1), and High (2). 

Since we want to predict the average rating of an recipe, we choose only features about the recipe's original qualities. Features we selected are the ones that are available at the “time of prediction”; that is, from the raw_recipe file. We choose average_rating as our response variable since it is a good indicator of the overall rating of a recipe. 

To evaluate model performance, overall accuracy and F1-score for each rating category were used as the primary metrics. Accuracy provides a general measure of how well the model performs across all predictions, while F1-score balances precision and recall, offering a better assessment for individual rating categories, especially when class distributions are imbalanced. Since the dataset is dominated by High ratings, relying solely on accuracy would be misleading, as the model could achieve high accuracy simply by predicting "High" most of the time. Therefore, the F1-score per class is essential to assess whether the model correctly identifies Medium and Low ratings, which are less frequent but still important.


## Baseline Model


## Final Model


## Fairness Analysis
