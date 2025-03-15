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
#### Data Cleaning
First, we will merge and edit the dataframe with the 4 steps:

1. **Left merge recipes and ratings dataset** - this helps us to put two dataframes together for further analysis. A left merge was done to keep all the rows in raw_recipes to ensure recipes without a user rating/comment is still included.
2. **fill all ratings of a value 0.0 with np.nan** - On food.com, star ratings range from 1 to 5. A rating value of 0.0 simply indicates that the user did not submit a star rating to that recipe, thus it should be replaced by NaN so we know it is a missing value.
3. **Find average rating per recipe** - Many recipes across the dataframe are rated more than once. This can be seen from the fact that interactions (user rating/comment) have a lot more rows than raw_recipes(one row per unique recipe). To better explore what could affect recipe ratings and to predict rating, we will be looking at the average rating per recipe.
4. **Add the average rating per recipe as a new column into the merged dataframe** - This steps creates an initial dataframe that we can use for our analysis


## Assessment of Missingness


## Hypothesis Testing
Despite exploring correlations, we want to test if there is a statistically significant difference between average ratings of recipes with different nutrient levels. The nutrient that we want to look at is calories.

Since the distribution of average ratings are not normal, instead skewed left with most average ratings between values of 4 to 5, we will do a permutation testing on the level of the nutrient calories and average rating of recipes.

* **Null Hypothesis**: High-calorie recipes (calorie ≥median calorie count in dataset) have the same ratings than low-calorie recipes (calories < median calorie count in dataset).

* **Alternative Hypothesis**: High-calorie recipes (≥median calorie count) have different average ratings than low-calorie recipes (<median calorie count).

We will be using a standard significance level of 0.05. The test statistic will be the difference of average ratings for high calorie recipes and the average ratings of low calorie recipes.

## Framing a Prediction Problem

We will train a **classification model** to predict **average ratings of recipes**. The baseline model used a Decision Tree Classifier with a depth limit of 5 to predict recipe ratings as High, Medium, or Low.  The target variable, rating_category, was **ordinal and encoded as Low (0), Medium (1), and High (2)**. 

Since we want to predict the average rating of an recipe, we choose only features about the recipe's original qualities. Features we selected are the ones that are available at the “time of prediction”; that is, from the raw_recipe file. We choose average_rating as our response variable since it is a good indicator of the overall rating of a recipe. 

To evaluate model performance, overall accuracy and F1-score for each rating category were used as the primary metrics. Accuracy provides a general measure of how well the model performs across all predictions, while F1-score balances precision and recall, offering a better assessment for individual rating categories, especially when class distributions are imbalanced. Since the dataset is dominated by High ratings, relying solely on accuracy would be misleading, as the model could achieve high accuracy simply by predicting "High" most of the time. Therefore, the F1-score per class is essential to assess whether the model correctly identifies Medium and Low ratings, which are less frequent but still important.


## Baseline Model

For our baseline model, we are utilizing a decision tree classifier and split the data points into training and test sets. The model included **seven quantitative features**: calories, total fat percentage daily value (PDV), sugar PDV, sodium PDV, protein PDV, saturated fat PDV, and carbohydrates PDV. Since all features were numerical, no nominal variables required one-hot encoding. To ensure a more stable and accurate model, numerical features were standardized using StandardScaler() before training.

The baseline model achieved an **overall accuracy** of 87.64%, but a closer look at the **F1-scores** revealed major shortcomings. The model performed well on High ratings (F1 = 0.93) but performed poorly on Medium (F1 = 0.02) and Low (F1 = 0.00). This means that while the model correctly classified most High ratings, it almost completely failed to distinguish Medium and Low ratings, likely because of severe class imbalance. As a result, while the accuracy appears high, the model is not useful for predicting lower-rated recipes, which limits its practical value.


## Final Model

For the final model, we use seven numerical features, unchanged from our baseline model, that are likely to impact predicted rating: calories, total fat percentage daily value (PDV), sugar PDV, sodium PDV, protein PDV, saturated fat PDV, and carbohydrates PDV. These features were chosen because they reflect healthiness, dietary preferences, and perceived recipe quality, which can all affect user satisfaction.

- "calories"

Users may favor recipes with a certain calorie range. Low-calorie recipes might be preferred by health-conscious individuals, while high-calorie recipes could be rated higher if they are perceived as more indulgent.

- "total_fat_PDV"

Fat content contributes to flavor and texture. Recipes with too little fat might be seen as bland, while excessive fat may lead to lower ratings from health-conscious users.

- "sugar_PDV"

Sweetness can enhance appeal, but excessive sugar might lead to lower ratings due to health concerns. This is particularly relevant for desserts vs. savory dishes.

- "sodium_PDV"

Saltiness can influence taste perception. While some sodium is essential for flavor, overly salty dishes may receive lower ratings.

- "protein PDV"

High-protein recipes may appeal to athletes and health-conscious users. Low-protein dishes may be perceived as lacking substance, affecting ratings.

- "saturated fat PDV"

While saturated fat contributes to taste and texture, excessive amounts may lead to lower ratings due to health concerns.

- "carbohydrates PDV"

Carbs contribute to satiety and energy content. Some users might prefer low-carb meals, while others might rate carb-heavy dishes higher based on their preferences.

For the final model, a **Random Forest Classifier** was used instead of a single decision tree. This change helped to reduce overfitting and improve generalization by averaging multiple decision trees trained on different subsets of the data. Additionally, **hyperparameter tuning** was performed using RandomizedSearchCV, optimizing key parameters such as number of trees (n_estimators), tree depth (max_depth), minimum samples required for splitting (min_samples_split), and minimum samples per leaf (min_samples_leaf). The final model was trained with 100 trees, a max depth of 20, min_samples_split=2, and min_samples_leaf=1, and it used class balancing (class_weight='balanced') to improve the classification of underrepresented classes.

The final Random Forest model achieved an **overall accuracy** of 91.39%, an improvement of 3.75 percentage points over the baseline model. More importantly, the **F1-score improved significantly across all categories**, with High increasing to 0.95, Medium improving to 0.55, and Low reaching 0.42. This indicates that the model was much better at distinguishing between different rating levels, particularly for Medium and Low ratings, which were previously misclassified almost entirely. These improvements confirm that the additional features helped capture important aspects of recipe complexity, and that Random Forest’s ability to handle imbalanced data improved classification performance across all rating categories.

## Fairness Analysis
