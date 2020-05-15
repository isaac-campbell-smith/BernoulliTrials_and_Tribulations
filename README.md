# Bernoulli Trials & Tribulations

#### Predicting fraud and anomaly detection on imbalanced classification data
Data source: https://www.cs.purdue.edu/commugrate/data/credit_card/
![Fraud Header]()

---
## Sections:
 |  **[Introduction](#introduction)**  |
 **[Data Cleaning & Exploration](#data-cleaning-&-exploration)**  |
 **[Dealing with Imbalanced Classifier](#dealing-with-imbalanced-classifier)**  |
 **[Cost Benefit & Scoring Metrics](#cost-benefit-&-scoring-metrics)**  |
 **[Focused Exploration](#focused-exploration)**  |
 **[Analysis](#analysis)**  |
 **[Future](#future)**  |<br><br>
 |  **[Takeaways](#takeaways)**  |
 
---
## Introduction
Fraud: boy is it a problem! Today we'll be looking at a 2009 dataset of anonymized credit card transaction data from a now defunct data-mining competition but currently available via the header link at cs.Purdue.edu
> The competition offers 2 version of the data - easy & hard mode. The hard mode purportedly offers several more powerful indicators but requires deeper cleaning. I went with hard mode. 
---
## Data Cleaning & Exploration
Original Dataset:
![Original Dataset](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/head.png)

![Original Describe](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/Describe.png)

The first thing t
Steps: remove transactions amounting to $0. Remove hour2 column (only a few rows not equal to hour1). State distribution doesn't look too crazy. There's one nan value on state so we'll assume input error and drop that row. Drop state and area code columns.

! [Fraud by State] (state bar)

More steps: Look at email anomalies. Of 67 email addresses making multiple transactions, only 5 were attributed solely to fraud (for a total of 11 fraudulent transactions). No reason to remove these for now. It seems there's a high outlier from ClintMiller.com. We'll remove that from our model and tell the IT dept. to just ban that domain for now. 

![Fraud by Email uniqueness] (scatter email)

More steps: We'll go ahead drop email column now as the account number is analgously unique. Finally look at all these weird fields and flags! Looks like Flag 3 isn't doing anything for us so we'll drop it....

Cleaned Dataset:
![Cleaned Dataset](clened pd df)

---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Dealing with an Imbalanced Classifier
My initial, admittedly naive, assumption about why predicting fraud in this data is so difficult is because of class imbalance. We only have 

> SMOTE sampling
Random undersampling of the majority class & oversampling of the minority class. 
The former is a straightforward concept but the latter has multiple approaches. You could simply sample with replacement, but SMOTING is a bit more sophisticated. One method is to oversample using a k-nearest neighbors algorithm at the BORDERLINE of minority/majority clusters. SVM is a similar technique, but instead recreates borderline samples using a Support Vector Machine algorithm. ADASYN (Adaptive Synthetic Sampling) was the most intriguing option to me because it recreates minority samples in majority class clusters. In practice, all these methods can lead to overfitting of the training set, however I did observe slight increases in prediction metrics on 

#### FEATURES
Better features make for better predictions. That isn't the most sophisticated statement but building a reliable model with this data is extremely challenging. 

Text goes here 

<br> 

> 

<br>


---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Cost Benefit & Scoring Metrics

Business problems require business solutions and while it would be great to just shoot for a perfect 100% accuracy, that's not super plausible. In the case of this dataset, it's likely impossible. Instead I'd like to explore a business scenario and the various cost/benefit outcomes of adopting vs. not adopting a model.

>*NOTE: The original competition guidelines specified results at a 20% lift. I'm not sure why. I went with ROC analysis instead.

#### The Scenario!
Our dataset consists of 99999 transactions over 98 days. This means we typically see about 1020 transactions per day, 26 of which are fraudulent. The typical sale is about $27.50 (fraudsters don't typically go much higher than ethical people). Let's assume an employee can review and determine whether 1 transaction was fraudulent in 45 minutes, or 10 per day. Of course there are companies with a more robust tech and employee infrastructures to do this more quickly -- we'll circle back to that.

Let's now imagine a world where we correctly flag all fraudulent cases and only fraudulent cases for review. This would require the labor of 3 employees, who we pay $20 an hour. This set of assumptions is problematic because we typically see 7 cases of fraud between 5pm and 8am vs 19 cases during working hours. Most people who have done online shopping before would not expect an order to process until the following business day, and since we typically see 2 cases of fraud per hour during work hours, it's reasonable to conclude that a 4 person fraud investigation team can keep up with the daily backlog with time leftover. Unfortunately though, nobody wants to work on weekends, and if they did, we'd have to pay them time and a half, making it more cost effective to just let fraud happen on weekends. This means we'll lose about $1430 to fraud per weekend and $1950 during a typical work week (or 97.5 employee hours). 

Under this story we've told for ourselves, incorrectly flagging fraud as not fraud costs us $27.50, while correctly flagging a legal transaction as fraud costs us $22.50 - $15.00 for employee time plus losing the opportunity to review actual fraud. If we correctly flag fraud we only lose $7.50 while correctly flagging legal transactions incurs no loss. 

---

<sub>[  **[Back to Sections](#sections)** ]</sub>

## Analysis
<br>


