# Bernoulli Trials & Tribulations

#### Predicting fraud and anomaly detection on imbalanced classification data with an unreasonable level of hatred for Kaggle data

Data source: https://www.cs.purdue.edu/commugrate/data/credit_card/

---
## Sections:
 |  **[Introduction](#introduction)**  |
 **[Data Cleaning & Exploration](#data-cleaning-&-exploration)**  |
 **[The Models](#the-models)**  |
 **[Cost Benefit & Scoring Metrics](#cost-benefit-&-scoring-metrics)**  |
 
---
## Introduction
Fraud: boy is it a problem! In the U.S. alone, fraud accounts for losses of about $170 Billion annually and the methods used by fraudsters are always changing. Thankfully, there are some pretty powerful machine learning algorithms that can be used to detect and prevent that from happening. Today we'll be looking at a 2009 dataset of anonymized credit card transaction data from a now defunct data-mining competition but currently available via the header link at cs.Purdue.edu
> The competition offers 2 version of the data - easy & hard mode. The hard mode purportedly offers several more powerful indicators but requires deeper cleaning. I went with hard mode. Both are more difficult to predict than the popular Kaggle fraud detection dataset that makes an appearance in many ML Youtube videos and I have hate in my heart for people that flex their 99.999% accuracy on it (which is not a good scoring metric to use by the way).

While the original hosting page of the data is no longer available, I managed to find a fraud detection model proposal that helped put a few things in context and provided inspiration along the way:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180893/

Here is a data snippet that illustrates the complexity of fraud detection. Can you see why?
![Similarties](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/FraudvsNot.png)

Rows 3 & 4 are identical column-wise, yet row 3 is classified as fraud and row 4 is not. Obviously, some fraudersters are highly skilled at circumventing detection by completely copying valid transactions. Perhaps this classification was owing to user error but I built my models knowing I wouldn't be able to perfectly classify all fraud cases. 

---
## Data Cleaning & Exploration
Original Dataset:
![Original Dataset](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/head.png)

![Original Describe](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Describe.png)

There's only one nan in a state cell so we'll just drop that row considering we have nearly 100,000 more transactions to work with. Interestingly there's a fair amount of transactions amounting to $0. My initial instinct was to remove these rows but my predictions got worse when I did - perhaps owing to phishing attempts to hack into an account (testing the waters essentially). I also removed hour2 as it mirrored hour1 in 98% of the data. 

![State Distributions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbystate.png)
![ZIP Distributions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbyzip.png)

These distributions seem to have a fair amount of overlap and including both does not seem sensible. I ended up using Zip in my final model training it's a more specific form of location. 

![Time Transactions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/transactionsbyhour.png)
![Time Fraud](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbyhour.png)

This is interesting! Fraud seems to happen on a greater proportion of off hours transactions but we'll still see a lot more fraud during the daytime.

![Email Unique](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/emailuniqueness.png)

As with $0 transactions, my intial impulse was to remove the outlier domain coming from ClintMiller.com (and theoretically tell IT to ban them) but ultimately found this led to weaker predictions. I also found that out of 67 accounts making multiple transactions, only 5 were attributed solely to fraud (for a total of 11 fraudulent transactions). This was another removal I thought might help but didn't. 

And finally here's a breakdown of all the different bonus features (I'm excluding Field3 because it's wild). It's reasonable to assume that at the very least all categories excluding field3, field4 and flag5 can be treated categorically rather than numerically. I one-hot-encoded field1 for training and did experiment with treating field4 and flag5 similarly though this only added hours to computation for 'barely-there' improvements to the output.
| Category       | Number of Unique Values     |
| :------------- | :----------: |
| field1     | 5 |
| field2     | 2 |
| flag1      | 2 |
| field3     | 16960 |
| field4     | 38 |
| indicator1 | 2 |
| indicator2 | 2 |
| flag2      | 2 |
| flag3      | 2 |
| flag4      | 2 |
| flag5      | 24 |


---
<sub>[  **[Back to Sections](#sections)** ]</sub>


## The Models
The original contest 'business problem' states that the goal here 'is to maximize accuracy of binary classification on a test data set, given a fully labeled training data set. The performance metric is the lift at 20% review rate'. I am instead choosing to look at Reciever Operating Characteristic curves(or ROC curves, the true positive against the false positive rate) for two reasons. One is that it's simply a more commonly used Classification metric. The other is that I wanted to explore a theoretical business scenario and the cost-benefit of adjusting classification probability thresholds. In some contexts, the optimal model will not maximize accuracy. 

### Logistic Regression
The classic classification algorithm! It takes some This model initially performed very poorly initially (essentially guessing badly) - it took some feature engineering to boost performance, but I wasn't expecting this model to perform as well as it did. 
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Logistic.png" width="550">

### Gradient Boosted Forests
#### XGBoost
XGBoost has been a very highly touted model in the last couple years and it did perform well, however, to get the ROC 5 points higher than a simple logistic regression came at a cost - it took almost 2 hours for this model to train (vs seconds for the logistic regression). Though xgboost does include a `DMatrix` datatype meant to dramatically speed up training, the documentation and stack overflow discussions on proper use is unclear, and I could not work past this error: `'DMatrix' object has no attribute 'shape'`.

After tuning some of the hyperparameters (see hyperparameter_tuning notebook), the output achieved here was trained with this model:
```
xgb = XGBClassifier(booster='dart', 
                    subsample=.8,
                    max_depth=8,
                    learning_rate=0.05, 
                    max_delta_step=0, 
                    min_child_weight=1, 
                    n_estimators=1500)
```

<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/XGBoost.png" width="550">


#### LightGBM 
LightGBM is similar to XGBoost with 2 key differences (in this work anyway). Firstly, it supports Scipy's sparse matrix without issue -- the above XGBoostClassifier ran for about 2 hours to fit all 5 K-folds while LightGBM took less than a minute.  Secondly, LightGBM incudes a `categorical_feature` parameter which allows you to specify training columns with `int` type as categories and the model handles one-hot-encoding under the hood. The results shown here labelled zip and customer id as categorical features, but otherwise identical to the XGBoostClassifier.
```
lgb = LGBMClassifier(boosting_type='dart', 
                     subsample=.8, subsample_freq=1,
                     categorical_features='2,3',
                     learning_rate=0.05, 
                     min_child_weight=1, 
                     n_estimators=1500)
```
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/LightGBM.png" width="550">


#### Isolation Forest
I also read a couple papers about Isolation Forests, which focuses on outlier detection. While it seems to be a potent algorithm, I did not find much success with it. It was much less consistent and accurate than any of the other models. The model tuning is a bit murky for me and the scale it assigns to the output is not a simple probability (rather -1 to 1). But I include it here for posterity.
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/IsoForest.png" width="550">

CatBoost was the model that seems to win the day here. What's great about this model is it's high ROC score, ease of use, and computation time. While it took a bit longer for this model to train than the Logistic Regression, it was remarkably faster than the XGB classifier (without any hyperparameter tuning!)
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Catboost.png" width="550">

I generated a holdout set to use as a final test for all the models just to verify there wasn't anything strange going on in the testing phase and these results did validate the testing observations.
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Compared.png" width="550">
<br>

---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Cost Benefit & Scoring Metrics

Business problems require business solutions and while it would be great to just shoot for a perfect 100% accuracy, that doesn't seem plausible here. When we're predicting things like fraud we have to take into consideration the cost of fraud, the cost of reviewing fraud, and the cost of hassling users with misclassifying legal transactions as fraud. 

### Different Business Scenarios
#### Base Case
Let's move beyond ROC curves and examine how our default CatBoost model does with calculating the probability of fraud on the test data.
You can see that we only correctly identify just over half of our fraud labels. Note that we do pretty well in avoiding false positives. The average transaction in this dataset is about $27 so we can expect the average cost of fraud to be about the same (it's actually lower here but let's assume it's the same). This dataset spans 98 days which gives me the impression they aren't making big enough money to efficiently review fraud. Let's say there are a few junior employees that take their time to review each case, specifically $15 in manhours. Let's assume there is no risk for user churn in misclassifying fraud, so it costs the same as reviewing fraud. Our loss minimizing probability threshold with this model under this scenario comes pretty close to the default 0.5 classification threshold (that is, all calculated fraud probabilities above 0.5 by the model in the testing data will be classifified as fraud). 
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/confusion_default.png" width="550">
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/default_profitcurve.png" width="516">

#### Low Cost Reviews
Now let's say we have the resources to very cheaply quickly review fraud and it only costs $2 in manhours. Again we assume that no risk for churn because of fraud reviews. Under this scenario, we can accept far more false positives while reducing the number of false negatives. The statistical terminology for this model is high recall, low precision.
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/confusion_low.png" width="551">
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/cheapreview_profitcurve.png" width="516">

#### High Churn Probability
Now let's say it costs about $5 to review fraud but we expect to lose $75 from user churn for misclassification. This is a bit of a ridiculous scenario, but I just want to illustrate how this changes the loss curve. Under this scenario we have to be more certain in our predictions and have to accept more fraud to occur to reduce false negatives (or low recall, high precision).
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/confusion_high.png" width="551">
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/churny_profitcurve.png" width="516">

![Cost]
---

<sub>[  **[Back to Sections](#sections)** ]</sub>

## Conclusion
With Machine Learning and Data Science being such trendy work these days, I hope this case study demystifies how it actually relates to doing business. As the previous section illustrates, ML algorithms can save businesses quite a bit of money, but it's almost equally as important to have a solid system for review in place. While CatBoost is currently my favorite algorithm after this exercise, the scoring metrics are close enough that other models are worthy of consideration.