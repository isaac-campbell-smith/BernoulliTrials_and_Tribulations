# Bernoulli Trials & Tribulations

#### Predicting fraud and anomaly detection on imbalanced classification data
Data source: https://www.cs.purdue.edu/commugrate/data/credit_card/
![Fraud Header]()

---
## Sections:
 |  **[Introduction](#introduction)**  |
 **[Data Cleaning & Exploration](#data-cleaning)**  |
 **[Dealing with Imbalanced Classifier](#initial-modeling)**  |
 **[Hypothesis](#hypothesis)**  |
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
### Graphs/Visualizations:
 |  **[Averge MPG by Manufacturer](#Average-MPG-by-Manufacturer)**  |
 **[Horsepower to MPG](#MPG-to-Horsepower-Scatter-Plot)**  |
 **[Torque to MPG](#MPG-to-Torque-Scatter-Plot)**  |
 **[Weight to MPG](#MPG-to-Weight-Scatter-Plot)**  |
 **[Passenger Door to MPG](#MPG-to-Passenger-Doors-Density-Plot)**  |<br>
<details>
  <summary>
    Show Graphs
  </summary>
<br>

> The rabbit hole of exploration
#### Average MPG by Manufacturer  
<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/avgmpgbrand.png" width="80%"></img>
<br>| **[Back](#graphsvisualizations)** |
#### MPG to Horsepower Scatter Plot
<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/mpgbyhp.png" width="80%"></img>
<br>| **[Back](#graphsvisualizations)** |
#### MPG to Torque Scatter Plot
<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/mpgbytq.png" width="80%"></img>
<br>| **[Back](#graphsvisualizations)** |
#### MPG to Weight Scatter Plot
<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/mpgbyweight.png" width="80%"></img>
<br>| **[Back](#graphsvisualizations)** |
#### MPG to Passenger Doors Density Plot
> Getting a little desperate to find something meaningful...
<img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/mpgbydoorsDensity.png" height="auto" width="80%"></img>
<br>| **[Back](#graphsvisualizations)** |
<br>


</details> 

Ever since the mishap of finding possibly corrupted data, as well as choosing a topic/hypothesis that was unsupportable or unquantifiable at this point (with my skillset), I started to delve deeper into the data. I knew I needed to visualize and compare different specifications against each other in order to find something interesting. 

<br> 

> With a 2020 mindset and considering the trajectory of car specifications over the years, I thought to myself, "*What are some important aspects that come to mind when people pick a car*? 

<br>

That's when it hit me! I started to focus my visualizations and comparisons over a car's MPG (Miles per Gallon). Over 10-20 plots later, and with the help of [Pandas-Profiling](https://github.com/pandas-profiling/pandas-profiling), I was finally able to reformulate a hypothesis.  

---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Focused Exploration

It was on this plot that I was pointed out that 5 Cylinder engines have a *Bimodal* distribution when it comes to their MPG

> #### MPG-to-n_Cylinders-Density-Plot <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/img/mpgbycyldense.png" width="80%"></img><br>

As you can see, with the graphs, there has been a progression on centering the focus of what exact pieces of data I wanted to further explore. 

<details>
  <summary>
    <b> Histogram of MPG of 5-CYL Engines with Dist </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/5cylmpgdist.png">
  </img>
</details>

<details>
  <summary>
    <b> MPG Dist of 5-CYL Engines by Manufacturer </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/5cylmpgbymodel.png">
  </img>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/5cylmpgbymodelgran.png">
  </img>
</details>

<details>
  <summary>
    <b> 5-CYL MPG Averages by Model  </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/mpgavg5cylmodel.png">
  </img>
</details>

---

<details>
  <summary>
    <b> 5-CYL Horsepower/Torque Averages by Model  </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/hptqavgmodel.png">
  </img>
</details>

<details>
  <summary>
    <b> 5-CYL Horsepower Densities by Manufacturer  </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/5cylhpmanufacturer.png">
  </img>
</details>

<details>
  <summary>
    <b> Power Densities by Cylinder  </b>  
  </summary>
  <img src="https://raw.githubusercontent.com/boogiedev/automotive-eda/master/focused_img/powdistbycyl.png">
  </img>
</details>

---
<sub>[  **[Back to Sections](#sections)** ]</sub>

## Hypothesis

After exploring the data, and thanks to the help of Tony and Andrew (my instructors) we saw that the MPG distribution of 5-Cylinder engines were Bimodal! This was interesting! Even having a bit of domain knowledge about cars, I couldn't exactly explain, or even prove why this was the case. 
<br>

> My initial thoughts: 

*5-Cylinder engine configurations are somewhat of a novelty engine, where the majority of car models who host this engine are coming from the same manufacturers who are grandfathering in this design choice as a statement of legacy, rather than practicality.*

<br>

**TLDR: 5-Cylinder Engine configurations offer _no advantages_ over more "Traditional" Engines (I-4, V-6)**

**Null :<br> 
![#f03c15](https://placehold.it/15/f03c15/000000?text=+) 5-Cyl Engine configs offer _no advantages_ over I-4, V-6 Engines configs in the categories below**<br>
**Alternative :<br>
![#c5f015](https://placehold.it/15/c5f015/000000?text=+) 5-Cyl Engine configs _do_ offer _advantages_ over I-4, V-6 Engines configs in the categories below**<br>

| Miles Per Gallon (MPG - Combined)  | Horsepower | Torque |
| ------------- | ------------- | ------------- |

<br>

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

