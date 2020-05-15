# Bernoulli Trials & Tribulations

#### Predicting fraud and anomaly detection on imbalanced classification data with an unreasonable level of hatred for Kaggle competitors

Data source: https://www.cs.purdue.edu/commugrate/data/credit_card/

---
## Sections:
 |  **[Introduction](#introduction)**  |
 **[Data Cleaning & Exploration](#data-cleaning-&-exploration)**  |
 **[Dealing with Imbalanced Classifier](#dealing-with-imbalanced-classifier)**  |
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

There's only one nan in a state cell so we'll just drop that row. Interestingly there's a fair amount of transactions amounting to $0. My initial instinct was to remove these rows but my predictions got worse when I did. I may do some more EDA on that group in the future but it seems that there are some powerful indicators of transactions at $0, probably owing to phishing attempts to hack into an account (testing the waters essentially). I also removed hour2 as it mirrored hour1 in 98% of the data. 

![State Distributions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbystate.png)
![ZIP Distributions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbyzip.png)

These distributions seem to have a fair amount of overlap and I didn't want both. I ended up using Zip in my final training features because I assume the more dramatic outliers would serve me better. Again, I may do the opposite later.

![Time Transactions](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/transactionsbyhour.png)
![Time Fraud](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/fraudbyhour.png)

This is interesting! Fraud seems to happen on a greater proportion of off hours transactions but we'll still see a lot more fraud during the daytime.

![Email Unique](https://raw.githubusercontent.com/isaac-campbell-smith/BernoulliTrials_and_Tribulations/master/visuals/emailuniqueness.png)

As with $0 transactions, my intial impulse was to remove the outlier domain coming from ClintMiller.com (and theoretically tell IT to ban them) but ultimately found this led to weaker predictions. I also found that out of 67 accounts making multiple transactions, only 5 were attributed solely to fraud (for a total of 11 fraudulent transactions). This was another removal I thought might help but didn't. 

And finally here's a breakdown of all the different bonus features (I'm excluding Field3 because it's wild). Each dictionary key is a 'category' and the values are the corresponding rate of fraud and occurences in the dataset. I separated out exclusively legal categories if they occur and only include the number of occurences. 

## >field1
FRAUD :
[{0: (0.02765, 10921)}, {3: (0.02568, 56885)}, {2: (0.02663, 25352)}, {1: (0.02484, 2737)}]

LEGAL :
[{4: 8}]

## >field2
FRAUD :
[{0: (0.02652, 57057)}, {1: (0.02556, 38846)}]

## >flag1
FRAUD :
[{0: (0.02498, 54808)}, {1: (0.02767, 41095)}]


## >field4
FRAUD :
[{19: (0.02315, 5055)}, {14: (0.02376, 2357)}, {23: (0.02639, 3524)}, {31: (0.00513, 195)}, {21: (0.02687, 4949)}, {24: (0.02814, 2843)}, {7: (0.02371, 7000)}, {9: (0.0281, 9750)}, {6: (0.02976, 5309)}, {10: (0.02961, 5302)}, {25: (0.02997, 2069)}, {17: (0.02268, 4674)}, {18: (0.0258, 5813)}, {16: (0.02494, 3970)}, {11: (0.02525, 3248)}, {20: (0.0264, 5417)}, {8: (0.02721, 8378)}, {22: (0.02424, 4497)}, {26: (0.02495, 1443)}, {32: (0.00694, 144)}, {29: (0.02963, 540)}, {13: (0.02505, 1836)}, {12: (0.02251, 2310)}, {27: (0.02726, 1064)}, {15: (0.02543, 2949)}, {28: (0.02183, 733)}, {30: (0.04063, 320)}, {35: (0.04545, 22)}, {33: (0.01515, 66)}, {37: (0.06667, 15)}, {34: (0.04762, 63)}, {36: (0.1, 20)}]

LEGAL :
[{40: 7}, {39: 5}, {38: 11}, {41: 2}, {42: 2}, {44: 1}]

## >indicator1
FRAUD :
[{0: (0.02634, 85053)}, {1: (0.02452, 10850)}]

## >indicator2
FRAUD :

[{0: (0.02617, 94262)}, {1: (0.02377, 1641)}]


## >flag2
FRAUD :
[{0: (0.02689, 42657)}, {1: (0.02552, 53246)}]


## >flag3
FRAUD :
[{0: (0.02613, 58717)}, {1: (0.02614, 37186)}]


## >flag4
FRAUD :
[{0: (0.02615, 94552)}, {1: (0.02443, 1351)}]
LEGAL | Category : group size
[]

## >flag5
FRAUD :
[{1: (0.02581, 80200)}, {2: (0.0284, 13942)}, {4: (0.01445, 346)}, {3: (0.0263, 1293)}, {8: (0.5, 2)}]

LEGAL :
[{5: 42}, {9: 1}, {1434: 5}, {7: 26}, {98: 2}, {3278: 13}, {0: 10}, {364: 1}, {6: 9}, {1600: 1}, {100: 9}, {1643: 1}]


---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Dealing with an Imbalanced Classifier
My initial, admittedly naive, assumption about why predicting fraud in this data is so difficult is because of class imbalance. We only have 

> SMOTE sampling
Random undersampling of the majority class & oversampling of the minority class. 
The former is a straightforward concept but the latter has multiple approaches. You could simply sample with replacement, but SMOTING is a bit more sophisticated. One method is to oversample using a k-nearest neighbors algorithm at the BORDERLINE of minority/majority clusters. SVM is a similar technique, but instead recreates borderline samples using a Support Vector Machine algorithm. ADASYN (Adaptive Synthetic Sampling) was the most intriguing option to me because it recreates minority samples in majority class clusters. In practice, all these methods can lead to overfitting of the training set and increase the variance though, however I did observe slight increases in mean prediction metrics so I ended up

#### FEATURES
Better features make for better predictions. That isn't the most sophisticated statement but building a reliable model with this data is extremely challenging because we can't say for sure what these variables are and how they interact without a lot of highly idividualized regression analysis (which I have not done a lot of!). I took 2 wildly different approaches to varying results.

1. OneHotEncode Field1. This seemed to me the most likely non-binary categorical feature. I applied a MinMaxScaler to Field3 to account for negative and zero values. I squared and took the log of field4 because it seemed to have a strong influence on classification and I think it needed to be mellowed out. I also took the log of zip, account ID, and the square root of amount.

2. OneHotEncode everything. Literally everything. This was super computationally expensive and I could only get it to run through an XGBoosting algorithm. 


---

## The Models
### Random & Isolated Forest Classifiers
Random Forests are great out of the box generally but they only performed OK in this case! Isolated Forests are a bit different because it uses unsupervised learning algorithms to look for outliers and scales it's probability from -1, 1. While I did find that this performed better than a normal Random Forest, I could not figure out how to use them in a 

### Apriori
I spent way more time than I am willing to admit on an incorrectly labeled dataset and my predictions were all horrible. Nothing was working. So I turned to the model that the aforementioned paper used in their experiment. The tl;dr of their method was to only train on accounts occuring multiple times so to establish a dataset of fraud transactions and legal transactions, look at a bunch of combinations of features with a specific level of support for that group, and classify testing data based on the max support between both groups. They reported a nearly 100% accuracy rate but at the cost of not even attempting to flag novel transactions.

### Logistic Regression


### XGBoost
This the model that seemed to perform the best overall on it's own and I took 2 very different approaches to using it. Given enough time or greater processing speed I could have been a bit more scientifically methodical with how I fed it features. 
On the untransformed data set it did pretty well. 
The main benefit of XGBoost over the traditional Gradient Boosting algorithm is computational speed (it's a lot faster). My understanding is that it benefits from working with OneHotEncoded sparse matrices, which was my first approach 




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
