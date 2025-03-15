# NutriRate: Predicting Recipe Ratings with Nutrition Data
Explore factors that could affect recipe rating, and predict ratings.

Authors: Fiona Zou, Ruofei(Alexandra) Mao

## Overview
This data science project conducted at UCSD explores the factors influencing recipe ratings using a dataset from [food.com](https://food.com). Specifically, this project investigates how various recipe nutrients, like calories and total fat, contributes to a recipe's rating. By analyzing these features, our project aims to explore viewer preferences of nutrition levels of highly-rated recipes and develop a predictive model for recipe ratings.

## Introduction
For the project, we will be using the a subset of recipes and ratings dataset from [food.com](https://food.com). First, we will load in the two datasets and explore their features. 
The dataframe `raw_recipes` contains **83782 rows**(each representing a unique recipe as indicated by the number of unique recipe IDs equals the number of total rows in this dataframe), and **12 columns** (features) describing each recipe. Specifically, the columns are:

1. **`name`** (object/string)- recipe names stored as text data;
2. **`id`'** (int)- numerical representation of recipe id;
3. **`minutes`** (int)- number of minutes to prepare the recipe;
4. **`contributor_id`** (int)- users' id who posted the recipe;
5. **`submitted`** (object/date) - date the recipe was submitted;
6. **`tags`** (object) - a list of strings representing tags for the recipe;
7. **`nutrition`** (object) - 7 nutrition information of the recipe, including #calories, total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV) in order of appearance; PDV is the abbreviation for 'percentage of daily value'.
8. **`n_steps`** (int) -number of steps in recipe;
9. **`n_steps`**(objects) - in-order text data for recipe steps;
10. **`description`** (object) -user descriptions of their recipe;
11. **`ingredients`** (object) -list of strings each representing a needed ingredient of recipe;
12. **`n_ingredients`** (int) -total number of recipe ingredients.

`interactions` contains a total of **731927 rows** of comments on recipes and its columns identifies features of each comment:

1. **`user_id`** (int) - unique numerical representation for each distinct user;
2. **`recipe_id`** - numerical representation of each distinct recipe, the same as `id` column in `raw_recipes`;
3. **`date`**(object) - date object containing the date comments are submitted;
4. **`rating`**(int) - integer (ranging from 0 to 5) of recipe rating submitted by the user;
5. **`review`**(object) - text data containing user reviews of recipes
   
After getting a brief overview, we brainstormed a list of questions that we are interested in for these datasets:

* What type of recipes (in terms of nutrition level) tend to have higher average ratings?
* What type of recipes (sweet or savory) tend to have higher average ratings?
* What is the relationship between total number of ingredients and average rating of recipes?
* Do certains users (with >= 20 comments submmited) have preferences to certain type of recipes that they tend to give higher ratings?
* Do newer recipes have higher average ratings than older recipes?
* Do recipes with certain ingredients (like chocolate, cheese, milk) tend to have a higher average rating than others?
  
We are interested to explore if users have preferences over, for example, healthier recipes that has a lower calorie level, or if users have preferences over sweeter recipes (higher sugar level), etc. So We decided to further explore the first question: ***What type of recipes (in terms of nutrition level) tend to have higher average ratings?***

By exploring the relationship of nutrition level and average rating of each distinct recipe, we can see if people who uses ['food.com'](https://food.com) have a preference over food types, flavor (eg. sweeter with higher sugar levels or less sweet), or healthy level (eg. trying to eat less calorie, or less carbohydrates). This can be important our audiences, as it will assist recipe contributors to learn about what their potential audiences prefer, and what they could dislike. Knowing the current trend or people's general favors provide helpful insights for recipe contributors to submit more recipes that fits more people's eating habits; and eventually this can be important for food.com users as they get to see more of their preferred food types recipes.



## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To make the two dataframes better fit our topic of exploration, we did the following steps for data cleaning:

1. **Left merge recipes and ratings dataset**:
   * this helps us to put two dataframes together for further analysis. A left merge was done to keep all the rows in raw_recipes to ensure recipes without a user rating/comment is still included.
2. **fill all ratings of a value 0.0 with np.nan**:
   * On food.com, star ratings range from 1 to 5. A rating value of 0.0 simply indicates that the user did not submit a star rating to that recipe, thus it should be replaced by NaN so we know it is a missing value.
3. **Find average rating per recipe**:
   * Many recipes across the dataframe are rated more than once. This can be seen from the fact that interactions (user rating/comment) have a lot more rows than raw_recipes(one row per unique recipe). To better explore what could affect recipe ratings and to predict rating, we will be looking at the average rating per recipe.
4. **Add the average rating per recipe as a new column into the merged dataframe**:
   * This steps creates an initial dataframe that we can use for our analysis
5. **Separate the 7 nutrients from 'nutrition' each into a new column**:
   * Since we are exploring if recipes with specific nutrition levels tend to have a higher average rating, we will further create and prepare a new dataframe from recipes_merged by separating each value in each row of the 'nutrition' column. we will also drop columns like 'review', 'n_steps', 'tags', etc. that are not related to nutrition level and user ratings.
6. **Create a new Dataframe with separated nutrient columns**:
   * This gives us a column that contains only the 7 nutrients, identification information like `contributor_id`, `recipe_id`, and `user_id`, and `rating` & `avg_rating`.
7. **Only leave unique recipes**:
   * There are recipes that received multiple comments/ratings, recipes that only had a few, and recipes that had none. Hence, we only keep one row per each distinct recipe, `ratings` is dropped and we shall use their `avg_rating` for further exploration. And since we are only looking at recipe ratings and nutrients for now, identification columns `user_id` and `contributor_id` will be dropped as well.

**First 5 rows of our final dataframe, `recipes_unique`, is shown below:**

```
| recipe_id | avg_rating | calories | total_fat | sugar | sodium | protein | sat_fat | carbs |
|-----------|-----------|----------|-----------|-------|--------|---------|---------|-------|
| 275022    | 3         | 386.1    | 34        | 7     | 24     | 41      | 62      | 8     |
| 275024    | 3         | 377.1    | 18        | 208   | 13     | 13      | 30      | 20    |
| 275026    | 3         | 326.6    | 30        | 12    | 27     | 37      | 51      | 5     |
| 275030    | 5         | 577.7    | 53        | 149   | 19     | 14      | 67      | 21    |
| 275032    | 5         | 386.9    | 0         | 347   | 0      | 1       | 0       | 33    |
```


### Exploratory Data Analysis
### Univariate
For univariate analysis, we looked at the distribution of all ratings, and the distribution of average rating of each distinct recipe:

<iframe
  src="assets/distrib_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/distrib_avg_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Most user comment left a rating of 4 stars or more. only few got 3 or less. Something interesting to notice is that there seems to be a slightly bit more of 1 star ratings than 2 star ratings. For average rating per unique recipe, most are also distributed at 5 and 4, or in-between these two. As for nutritions, the distribution of each nutrition seems to be mostly compact, with a few very extreme outliers (for example, a recipe with 30k calories compared to the common value of way less than 10k). This outliers could signify an uncommon recipe. 

### Bivariate
For bivariate analysis, we will separately plot the trend of each nutrient vs. the average recipe rating, with avg_rating on the x-axis and nutrients on the y-axis:

<iframe
  src="assets/calories.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/total_fat.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/sugar.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/sodium.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/protein.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/sat_fat.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/carbs.html"
  width="1000"
  height="600"
  frameborder="0"
></iframe>

There appears to be a little correlation between each nutrients and recipe average rating. For example, carbohydrate, protein, sugar, and calories level tends to go down as average rating of recipes geting higher and closer to 5, while other nutrients tends to go up. However, they don't really have a very visually strong correlation: taking carbohydrate level as an example, it tends to go down as rating gets higher, but there are still a lot of recipes with a rating of 4 or higher that has a high carbohydrate level.

**To see the correlations clearer, we did a correlation heat map:**

<iframe
  src="assets/corr_heatmap.html"
  width="800"
  height="800"
  frameborder="0"
></iframe>

from the correlation heat map above, we can see that not really any of the nutrients have a correlation r with avg_rating that is more than + or 0.05, suggesting the correlations are not storng enough by one nutrients. However, we can see that some nutrient pairs have higher correlations. Specifically, **fat & calories**, **fat & saturated fat**, **protein & calories** all have a correlation >= 0.7. We will pair these up in aggregations and see if it changes anything.

### Intersting Aggregates
Below shows the pivot table of average rating of each of the three highly correlated pairs mentioned above, each separated to levels 'low', 'medium', and 'high':



<div style="margin-top: -50px; margin-bottom: -50px;">
    <iframe src="assets/proteinvscalories.html" width="800" height="300"></iframe>
</div>

<div style="margin-top: -50px; margin-bottom: -50px;">
    <iframe src="assets/total_fatvscalories.html" width="800" height="300"></iframe>
</div>

<div style="margin-top: -50px; margin-bottom: -50px;">
    <iframe src="assets/total_fatvssat_fat.html" width="800" height="300"></iframe>
</div>




The dark blue sections of each pivot table shows the highest average rating obtained in a certain pair combination column. It is interesting to notice that low protein and high total fat seems to obtain higher ratings, while calories and saturated fat doesn't show a much significant result of correlation with recipe rating.


## Assessment of Missingness
There are seven columns that contains missing values: `name`, `user_id`, `date`, `description`, `rating`, `avg_rating` and `review`. Specifically, the first three only contains one NaN values each. 

We already know, for `rating` and `avg_rating`, the NaN values implies ratings of 0.0, or that the user simply did not leave a star rating. Similarly, we suspect missingness in `review` and `description` are users not leaving a comment, and contributor not uploading a recipe description. **It is possible that these columns are NMAR.** For example, empty description may have some dependency on how many complicated a certain recipe is, which can be seen from number of steps and ingredients required. These will be explored later in thsi section. 

Since the dataset is directly generated by webscraping from food.com, there could be some algorithmic error during the scarping process that leads to one single missingness columns name, user_id, and date. So we will not specifically explore potential NMAR possibilities for these columns, since they only have 1 NaN values, and conclusions can't be drawn without further information of the web-scraping process.

**We picked the column `description` that could be NMAR to further explore its missing mechanism.** It's missingness could depend on the complexity of recipes. For example, an easy recipe with little steps or ingredients is pretty self-explanatory without the need for a description. Or reversely, it is difficult to summarize a very complex recipe so sometimes they are left blank. 

A permutation tests were performed by shuffling **`n_ingredients`** 1000 times to create 1000 simulations of the mean difference between number of ingredients of recipes with and without a description.
**Null Hypothesis**: The missingness of recipe description does not depend on number of ingredients of the recipe.
**Alternative Hypothesis**: The missingness of recipe description does depend on number of ingredients of the recipe.
**Test Statistic**: mean difference between the number of ingredients of recipes with and without a description. 
**Significance Level**: 0.05

<iframe
  src="assets/perm_ingredients.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
With a p-value of 0.002 < 0.05, we reject the null hypothesis. Missingness of `description` has dependency on `n_ingredients`.


A second permutation tests were performed by shuffling **`n_steps`** 1000 times to create 1000 simulations of the mean difference between number of steps of recipes with and without a description.
**Null Hypothesis**: The missingness of recipe description does not depend on number of steps of the recipe.
**Alternative Hypothesis**: The missingness of recipe description does depend on number of steps of the recipe.
**Test Statistic**: mean difference between the number of steps of recipes with and without a description. 
**Significance Level**: 0.05
<iframe
  src="assets/perm_steps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

With a p-value of 0.252 > 0.05, we fail to reject the null hypothesis. We cannot conclude that missingness of `description` has a dependency on `n_steps`.


## Hypothesis Testing
Many people are aware and try to control #calories they intake each day, so despite calories not showing a graphically significant correlation to recipe rating, we are still curious to test if there is a statistically significant difference between average ratings of recipes with different calories levels, or if calories doesn't actually affect ratings that much.

Since the distribution of average ratings are not normal, instead skewed left with most average ratings between values of 4 to 5, we will do a permutation testing on the level of the nutrient calories and average rating of recipes.

To do this, we found the mean calories across all recipes in `recipes_unique`, and we split the recipes into two group: high- recipes with calories >= median calorie, and low- recipes with calories < median calorie.

* **Null Hypothesis**: High-calorie recipes (calorie ≥median calorie count in dataset) have the same ratings than low-calorie recipes (calories < median calorie count in dataset).
* **Alternative Hypothesis**: High-calorie recipes (≥median calorie count) have different average ratings than low-calorie recipes (<median calorie count).
* **Test Statistic**: mean differences between recipe average rating of recipes with a calorie level higher than the median calories and recipes with calorie levels lower than the median calories.
* **Significance Level**: 0.05

<iframe
  src="assets/perm_hyp.html"
  width="800"
  height="800"
  frameborder="0"
></iframe>

Since the p-value is **0.0683** (and it's always somewhere around 0.06 in the multiple times we ran the permutation test), we obtained a p-value > 0.05. Thus, we **fail to reject the null hypothesis** of "High-calorie recipes (calorie ≥median calorie count in dataset) have the same ratings than low-calorie recipes (calories < median calorie count in dataset)" from this permutation test. We don't have enough evidence to suggest that people rate recipes differently based on if the recipe has a higher or lower calories level. Despite somewhat counter-intuitive, #calories of recipes does not show affect to the ratings of recipes so far.

## Framing a Prediction Problem

We will train a **multiclass classification model** to predict **average ratings of recipes**. We choose average ratings of recipes since it is a good indicator of the overall rating of a recipe. The baseline model used a Decision Tree Classifier with a depth limit of 5 to predict recipe ratings as High, Medium, or Low. The target variable, rating_category, was **ordinal and encoded as Low (0), Medium (1), and High (2)**. 

Since we want to predict the average rating of an recipe, we choose only features about the recipe's original qualities. Features we selected are the ones that are available at the “time of prediction”; that is, from the raw_recipe file. We choose average_rating as our response variable since it is a good indicator of the overall rating of a recipe. 

To evaluate model performance, **overall accuracy and F1-score for each rating category** were used as the primary metrics. Accuracy provides a general measure of how well the model performs across all predictions, while F1-score balances precision and recall, offering a better assessment for individual rating categories, especially when class distributions are imbalanced. Since the dataset is dominated by High ratings, relying solely on accuracy would be misleading, as the model could achieve high accuracy simply by predicting "High" most of the time. Therefore, the F1-score per class is essential to assess whether the model correctly identifies Medium and Low ratings, which are less frequent but still important.


## Baseline Model

For our baseline model, we are utilizing a **decision tree classifier** and split the data points into training and test sets. The model included **seven quantitative features**: calories, total fat percentage daily value (PDV), sugar PDV, sodium PDV, protein PDV, saturated fat PDV, and carbohydrates PDV. Since all features were numerical, no nominal variables required one-hot encoding. To ensure a more stable and accurate model, numerical features were standardized using **StandardScaler()** before training.

The baseline model achieved an **overall accuracy of 87.64%**, but a closer look at the **F1-scores** revealed major shortcomings. The model performed well on High ratings (F1 = 0.93) but performed poorly on Medium (F1 = 0.02) and Low (F1 = 0.00). This means that while the model correctly classified most High ratings, it almost completely failed to distinguish Medium and Low ratings, likely because of severe class imbalance. As a result, while the accuracy appears high, the model is not useful for predicting lower-rated recipes, which limits its practical value.


## Final Model

For the final model, we use nine numerical features, **with two newly engineered features**, that are likely to impact predicted rating: calories, total fat percentage daily value (PDV), sugar PDV, sodium PDV, protein PDV, saturated fat PDV, carbohydrates PDV, **nutrition score**, and **caloric density**. These features were chosen because they reflect healthiness, dietary preferences, and perceived recipe quality, which can all affect user satisfaction.

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

- **"nutrient_score"** (New Feature)

This feature balances positive and negative nutritional attributes. It is calculated as: nutrient_score = protein_PDV − (sugar_PDV + saturated_fat_PDV + sodium_PDV)/3
 
This ensures that recipes high in protein (a desirable trait for many users) score higher, while those with excessive sugar, saturated fat, or sodium (potential health concerns) score lower. Helps measure overall recipe healthiness by rewarding protein and penalizing unhealthy ingredients like sugar, sodium, and saturated fat.

- **"caloric_density"** (New Feature)

This feature measures calories per unit of macronutrient content, helping to distinguish between energy-dense and nutrient-dense recipes. It is calculated as: caloric_density = calories / (carbohydrates_PDV + protein_PDV + total_fat_PDV). Helps distinguish energy-dense vs. nutrient-dense foods, which impacts user preferences.

For the final model, a **Random Forest Classifier** was used instead of a single decision tree. This change helped to reduce overfitting and improve generalization by averaging multiple decision trees trained on different subsets of the data. Additionally, **hyperparameter tuning** was performed using RandomizedSearchCV, optimizing key parameters such as number of trees (n_estimators), tree depth (max_depth), minimum samples required for splitting (min_samples_split), and minimum samples per leaf (min_samples_leaf). The final model was trained with 200 trees, a max depth of 20, min_samples_split = 2, and min_samples_leaf = 1, and it used class balancing (class_weight='balanced') to improve the classification of underrepresented classes.

The final Random Forest model achieved an overall accuracy of **92.86%**, an improvement over the baseline model. More importantly, the **F1-score improved significantly across all categories**, with High increasing to 0.96, Medium improving to 0.67, and Low reaching 0.54. This indicates that the model was much better at distinguishing between different rating levels, particularly for Medium and Low ratings, which previously had lower recall and precision. These improvements confirm that Random Forest’s ability to handle imbalanced data improved classification performance across all rating categories.

The addition of nutrient_score and caloric_density contributed to these gains by helping the model better understand how nutritional factors influence user ratings. These engineered features likely captured nuances that were not as apparent in the raw nutritional values alone, reinforcing the importance of feature engineering in improving model performance.

## Fairness Analysis

For this fairness analysis, we compare the model’s performance for two groups based on **caloric content**. Group X (Low-Calorie Recipes) consists of recipes with calories ≤ median(calories), while Group Y (High-Calorie Recipes) includes recipes with calories > median(calories). We selected calorie content as the distinguishing factor because it plays a significant role in user preferences and ratings. Some users may rate low-calorie meals lower due to taste, while others might rate high-calorie meals lower due to health concerns. If our model exhibits bias, it could perform better at classifying one group’s ratings more accurately than the other, affecting its fairness in prediction.

**Null Hypothesis**: Our model is fair. Its precision for recipes with higher calories and lower calories are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: Our model is unfair. Its precision for recipes with lower calories is lower than its precision for recipes with higher calories.

**Test Statistic**: difference in precision scores for High ratings between the two groups

**Significance Level**: 0.05

To measure fairness, we use **Precision for High ratings** ("High" = 2) as the evaluation metric. Precision is an appropriate choice because it assesses how many recipes predicted to be High are actually High-rated, ensuring that the model is not making excessive false positive errors for one group over the other. A lower precision score for one group would indicate that the model is more prone to incorrectly assigning High ratings to that group, suggesting potential bias.


<iframe
  src="https://aaarf404.github.io/Recipe-Rating-Prediction/assets/permutation1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


After running the permutation test, we obtained an observed precision difference of 0.029, meaning the model's precision for High ratings is 2.8 percentage points lower for Low-Calorie recipes compared to High-Calorie recipes. The computed p-value is 0.0, meaning that none of the 10,000 randomly shuffled precision differences were as extreme as the observed difference (0.029).

Since p-value < 0.05, we reject the null hypothesis. This means that the difference in precision between Low-Calorie and High-Calorie recipes is statistically significant and unlikely to have occurred due to random chance. In other words, our model performs significantly worse for Low-Calorie recipes in terms of precision for predicting "High" ratings.
