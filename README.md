# Bank Deposit Prediction

<br />
<br />

## Executive Summary
In this project, we built a classification model that effectively predicts term deposit customers. Term deposits are an important source of bank capital, and banks that more effectively target prospective term deposit customers are more likely to thrive. Applying modern machine learning techniques, our Light Gradient Boosting model vastly outperforms the status quo results where no model was used. Without our model, the success rate of acquiring term deposit customers is 12%, and with our model the success rate increases up to 5 times to 60%. We strongly believe our model generates significant business values for the bank, increasing its operational efficiency, reducing wastage, and eventually improving profitability and shareholder value.

<br />

-----------------------------------------------------------------------------
<br />

## Table of Contents
### Part I: Background Information (README file)
#### 1. The Business Problem
#### 2. The Goal
#### 3. The Evaluation Metrics
#### 4. Definition of Success
#### 5. Data Description
#### 6. Libraries and Version

### Part II: Data Exploration, Modelling and Evaluation (ipynb file)
#### 1. Setting up the work environment
#### 2. Importing the data set
#### 3. Exploratory Data Analysis (EDA)
#### 4. Baseline Modelling
#### 5. Feature Engineering and Modelling
#### 6. Final Model Building and Hyperparameter Tuning
#### 7. Model Evaluation and Business Value
#### 8. Limitations
#### 9. Conclusion


<br />

-----------------------------------------------------------------------------
<br />

### Part I: Background Information
### 1. The Business Problem
**This project builds a classification model for predicting marketing outcomes** using data from a Portuguese bank. The bank conducts telemarketing compaigns to encourage customers to open a term deposit. The outcomes from previous campaigns were collected, and a detailed description of the data set is included in section 5 "Data Description".


**Why do banks want term deposits in the first place?** 

If a bank wants to expand its business, for example by lending more money out, it will need to raise capital, and deposits are an essential (and cheap) source of capital.


**How can banks increase term deposit customers?** 

A brute force method is to contact every customer in the population and see how many will become depositers. In reality, businesses are constrained by limited resources. In the short term, the bank only has a fixed number of sales staff, therefore in order to increase customer coverage, the bank will have to recruit more sales people, provide additional training, rent larger offices, and invest in more infrastructures. Unfortunately, these investments in time and money do not guarantee business success, meaning that brute force expansions can be both costly and risky.

A better approach is to recognize existing limitations and shift the focus to improving business efficiency. More precisely, the bank should seek to answer the question of "How can we improve the success rate of obtaining deposit customers at current resource levels?". 

A visual comparison of the two above approaches is illustrated below.
<img src="Images/Comparison of two approaches.png">



<br />

-----------------------------------------------------------------------------
<br />


### 2. The Goal
Our goal is to build a classification model that effectively predicts which customers are more likely to open a term deposit. These predictions will allow the bank to focus resources primarily on the likely customers, thereby increasing the success rate of the marketing campaigns. The business value comes from the fact that higher success rate tends to promote efficiency, reduce waste and costs, and improve profitability.


<br />

-----------------------------------------------------------------------------
<br />


### 3. The Evaluation Metrics 
Common evaluation metrics for classfication models include: accuracy, precision, recall, F1-score, and lift/gain charts. There is no single metric that fits all problems. 

In the case of the bank marketing data set, we have imbalanced target variable classes, and therefore accuracy will not be a suitable metric. Precision indicates how many of the predicted deposit customers actually open a loan. While recall says of all the customers that open a loan, how many of them does our model catch? Intuitively, both precision and recall matter in this context, however we tend to prefer a single number metric for model evaluation. This is especially true as there exists a trade-off between precision and recall (for example, higher precision generally means lower recall and vice versa), making it difficult to consistently compare models based on either metric. F1-score is the harmonic mean of precision and recall, and gives us a good single number metric for model evaluation.

While the F1-score is a convenient and useful metric during the modelling stage, it does come with the disadvantage of not providing an intuitive interpretation. Therefore, at the business value evaluation stage, we will introduce the lift chart, which evaluates model performance by directly comparing business outcomes obtained with and without the model.

<br />

-----------------------------------------------------------------------------
<br />


### 4. Definition of Success
Our benchmark telemarketing success rate based on raw data is 12% (calculated as $\frac{Number \space of \space customers \space who \space open \space deposits}{Total \space number \space of \space customers \space contacted}$).

Our model will be considered successful, if it is able to achieve at least a 24% success rate. 

(Note this is an arbitrary value, but it is chosen to recognize the fact that in real business settings, often a sizeable improvement is needed to justify the significant investments and risks associated with business changes).


<br />

-----------------------------------------------------------------------------
<br />

### 5. Data Description

**Data Sources**

Originally created by: 
Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL)

The data set was fully described in:
S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.
In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães,
Portugal, October, 2011. EUROSIS.

Downloaded from:
https://www.kaggle.com/edith2021/bank-marketing-campaign

**Data Set Description**

This is a real data set from a Portuguese banking institution. This data set records the outcomes of telephone marketing campaigns for obtaining new term deposit customers. The data contains no sensitive information or customer identifiers to protect customer privacy.

Number of Observations: 45211.

Number of Variables: 17 variables (16 input and 1 target).

**Variable Description**

Input Variables:

1 - age (numeric)

2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
"blue-collar","self-employed","retired","technician","services")

3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

4 - education (categorical: "unknown","secondary","primary","tertiary")

5 - default: has credit in default? (binary: "yes","no")

6 - balance: average yearly balance, in euros (numeric)

7 - housing: has housing loan? (binary: "yes","no")

8 - loan: has personal loan? (binary: "yes","no")

9 - contact: contact communication type (categorical: "unknown","telephone","cellular")

10 - day: last contact day of the month (numeric)

11 - month: last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")

12 - duration: last contact duration, in seconds (numeric)

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

15 - previous: number of contacts performed before this campaign and for this client (numeric)

16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Target Variable
17 - y - has the client subscribed a term deposit? (binary: "yes","no")

<br />

-----------------------------------------------------------------------------
<br />

### 6. Libraries and Version
- Numpy:       1.19.5
- Pandas:      1.4.0
- Matplotlib:  3.5.1
- Seaborn:     0.11.2
- Sklearn:     0.23.2
- Pycaret:     2.3.6
