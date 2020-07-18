# Bernoulli Trials & Tribulations

#### Predicting fraud and anomaly detection on imbalanced classification data with an unreasonable level of hatred for Kaggle competitors

Data source: https://www.cs.purdue.edu/commugrate/data/credit_card/

---
## Sections:
 |  **[Introduction](#introduction)**  |
 **[Data Cleaning & Exploration](#data-cleaning-&-exploration)**  |
 **[Cost Benefit & Scoring Metrics](#cost-benefit-&-scoring-metrics)**  |
 **[The Models](#the-models)**  |
 **[Analysis](#analysis)**  |
 |  **[Takeaways](#takeaways)**  |
 
---
## Introduction
Fraud: boy is it a problem! Today we'll be looking at a 2009 dataset of anonymized credit card transaction data from a now defunct data-mining competition but currently available via the header link at cs.Purdue.edu
> The competition offers 2 version of the data - easy & hard mode. The hard mode purportedly offers several more powerful indicators but requires deeper cleaning. I went with hard mode. Both are more difficult to predict than the popular Kaggle fraud detection dataset everybody seems content to work with and I have hate in my heart for people that flex on it.

While the original hosting page of the data is no longer available, I managed to find a fraud detection model proposal that helped put a few things in context and provided inspiration along the way:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180893/

Here is a data snippet that illustrates the complexity of fraud detection. Can you see why?
![Similarties](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/FraudvsNot.png)

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

And finally here's a breakdown of all the different bonus features (I'm excluding Field3 because it's wild). It's reasonable to assume that at the very least all categories excluding field3, field4 and flag5 can be treated categorically rather than numerically. I one-hot-encoded field1 for training and did experiment with treating field4 and flag5 similarly though this only added hours to computation for minute changes to the output.
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

### Logistic Regression
The classic classification algorithm! It takes some This model initially performed very poorly initially (essentially guessing badly) - it took some feature engineering to boost performance. I wasn't expecting this model to perform as well as it did. 
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Logistic.png" width="550">

### Gradient Boosted Random Forests
I explored several variants of this type of model. XGBoost has been very highly touted and it did do well, however, to get the ROC 5 points higher than a simple logistic regression came at a cost - it took almost 2 hours for this model to train (vs seconds for the logistic regression). 
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/XGBoost.png" width="550">
<br>
I also read a couple papers about Isolation Forests, which focuses on outlier detection. While it seems to be a potent algorithm, I did not find much success with it. It was much less consistent and accurate than any of the other models. The model tuning is a bit murky for me and the scale it assigns to the output is not a simple probability. But I include it here for posterity.
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/IsoForest.png" width="550">
<br>
CatBoost was the model that seems to win the day here. What's great about this model is it's high ROC score, ease of use, and computation time. While it took a bit longer for this model to train than the Logistic Regression, it was remarkably faster than the XGB classifier (without any hyperparameter tuning!)
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Catboost.png" width="550">
<br>
I generated a holdout set to use as a final test for all the models just to verify there wasn't anything strange going on in the testing phase and these results did validate the testing observations.
<br><br>
<img src="https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Compared.png" width="550">
<br>

### Apriori
I spent way more time than I am willing to admit on an incorrectly labeled dataset and my predictions were all horrible. Nothing was working. So I turned to the model that the aforementioned paper used in their experiment. The tl;dr of their method was to only train on accounts occuring multiple times so to establish a dataset of fraud transactions and legal transactions, look at a bunch of combinations of features with a specific level of support for that group, and classify testing data based on the max support between both groups. They reported a nearly 100% accuracy rate but at the cost of not even attempting to flag novel transactions.

## Analysis
![ROC Curves]()
Parameters!
---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Cost Benefit & Scoring Metrics

Business problems require business solutions and while it would be great to just shoot for a perfect 100% accuracy, that's not super plausible. In the case of this dataset, it's likely impossible. Instead I'd like to explore a business scenario and the various cost/benefit outcomes of adopting vs. not adopting a model.

>*NOTE: The original competition guidelines specified results at a 20% lift. I'd imagine this is because 11 years ago, implementing a model was a lot more computationally expensive so they were looking for a strong confidence interval around a subgroup. I went with traditional ROC metrics to evaluate my models' performance.

#### The Scenario!
Our dataset consists of 99999 transactions over 98 days. This means we typically see about 1020 transactions per day, 26 of which are fraudulent. The typical transaction is about $27.50 (interestingly, fraudsters typically pull out only $23.00). Let's assume an employee can review and determine whether 1 transaction was fraudulent in 45 minutes, or 10 per day. Of course there are companies with a more robust tech and employee infrastructures to do this more quickly -- we'll circle back to that.

Let's now imagine a world where we correctly flag all fraudulent cases and only fraudulent cases for review. This would require the labor of 3 employees, who we pay $18 an hour. This set of assumptions is problematic because we typically see 7 cases of fraud between 5pm and 8am vs 19 cases during working hours. Most people who have done online shopping before would not expect an order to process until the following business day, and since we typically see 2 cases of fraud per hour during work hours, it's reasonable to conclude that a 4 person fraud investigation team can keep up with the daily backlog with time leftover. Unfortunately though, nobody wants to work on weekends, and if they did, we'd have to pay them time and a half, making it more cost effective to just let fraud happen on weekends. This means we'll lose about $1,196.00 to fraud per weekend and $1,950 during a typical work week (or 97.5 employee hours). 

Under this story we've told for ourselves, incorrectly flagging fraud as not fraud costs us $23.00, while incorrectly flagging a legal transaction as fraud costs us $18.00 -- 13.5 for employee time + the opportunity loss of reviewing actual fraud. If we correctly flag fraud we only have our $13.50 employee time while correctly flagging legal transactions incurs no loss. 

There's a great irony to me in top-level fraud detection work. At high end firms where the cost of reviewing false positives is probably going to be much lower, you can afford to skew your predictions towards a perfect recall with a lot of false positives. 

![Cost]
---

<sub>[  **[Back to Sections](#sections)** ]</sub>




## Takeaways
