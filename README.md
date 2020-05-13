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
Fraud: boy is it a problem! Today we'll be looking at a 2009 dataset of anonymized e-commerce transaction data from a now defunct data-mining competition but currently available via the header link at cs.Purdue.edu
> The competition offers 2 version of the data - easy & hard mode. The hard mode purportedly offers several more powerful indicators but requires deeper cleaning. I went with hard mode. 
---
## Data Cleaning & Exploration
Original Dataset:
![Original Dataset](My pd df)
Steps: remove transactions amounting to $0. Remove hour2 column (only a few rows not equal to hour1). State distribution doesn't look too crazy. There's one nan value on state so we'll assume input error and drop that row. Drop state and area code columns.

! [Fraud by State] (state bar)

More steps: Look at email anomalies. Of 67 email addresses making multiple transactions, only 5 were attributed solely to fraud (for a total of 11 fraudulent transactions). No reason to remove these for now. It seems there's a high outlier from ClintMiller.com. We'll remove that from our model and tell the IT dept. to just ban that domain for now. 

![Fraud by Email uniqueness] (scatter email)

More steps: We'll go ahead drop email column now as the account number is analgously unique. Finally look at all these weird fields and flags! Looks like Flag 3 isn't doing anything for us so we'll drop it....

Cleaned Dataset:
![Cleaned Dataset](clened pd df)

---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Dealing with Imbalanced Classifier


> SMOTE sampling


</details> 

Text goes here

<br> 

> 

<br>


---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Cost Benefit & Scoring Metrics

Business problems require business solutions and while it would be great to just shoot for a perfect 100% accuracy, that's not super plausible. Instead we'll 

>*NOTE: The original competition guidelines specified results at a 20% lift. This isn't a great metric so I went with my own.


---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Analysis
<br>

After attempting to find the correct sort of statistical analysis in this situation, I ended up trying to bootstrap data given by the 5-CYL Dataset. I realized that I ended up going in the wrong direction with all of this, and am showing only a slice of the work that was committed towards analysis. 

### MPG vs Cylinder Testing  <br>

<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/5v4mpgdist.png">

```python
# Five Cylinder Car 5 Num Sum for MPG
five_cyl_desc = five_cyl_cars["Gas Mileage (Combined)"].describe()
five_cyl_desc
```
<details>
  <summary>
    Output
  </summary>
  
| count | 528.000000 |
| ------------- | ------------- |
| mean | 22.696970 |
| std | 2.969809 |
| min | 15.000000 |
| 25% | 20.000000 |
| 50% | 23.000000 |
| 75% | 25.000000 |
| max | 27.500000 |
| Name: Gas Mileage (Combined), dtype: float64 |
  
  
</details>

```python
# Four Cylinder Car 5 Num Sum for MPG
four_cyl_desc = four_cyl_cars["Gas Mileage (Combined)"].describe()
four_cyl_desc
```
<details>
  <summary>
    Output
  </summary>
  
| count | 10107.000000 |
| ------------- | ------------- |
| mean | 27.139705 |
| std | 5.051244 |
| min | 17.000000 |
| 25% | 24.000000 |
| 50% | 26.000000 |
| 75% | 29.000000 |
| max | 58.500000 |
| Name: Gas Mileage (Combined), dtype: float64 |


</details>

<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/4v5bootstrap.png">


---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Future
- Implement more Hypothesis Testing on other categories in Engine output -> Bonferroni Correction
- Actually Source 0-60 Data or Clean existing
- Source Data on Engine Dynamics
---


## Takeaways
Thoughts:
- MM

