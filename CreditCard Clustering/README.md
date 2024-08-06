Dataset : [Credit-Card](./Adhy_Arya.csv)  
Object  : Credit Card depiction of data clustering concept using Scikit-Learn 

## Problem Statement ##
Analyze and cluster credit card holders based on their transaction data to understand customer segmentation and behavior.

## Background
The dataset comprises approximately 4,475 records containing information on purchases, credit limits, payments, and balances for credit card accounts. The goal is to apply clustering methods to discern the distribution and characteristics of different customer groups. This analysis will help in understanding customer behavior and improving credit card product offerings.

## Analysis
The analysis of balance and balance frequency revealed four distinct clusters. Cluster 0 consists of customers with high balances and purchase frequency, who make frequent installments but do not often update their balances. Cluster 1 is characterized by customers with low balances, purchase frequency, and installments, but who frequently update their balances. Cluster 2 includes customers with low purchasing power and infrequent balance updates, while Cluster 3 has customers with low purchasing power but who frequently update their balances.

When clustering by purchases and payments, it was observed that most customers belong to Cluster 3, with purchase amounts ranging from 3,361 to 41,000. Cluster 0 rarely makes purchases, while Clusters 1 and 2 exhibit low activity in making purchases. This led to the conclusion that Clusters 0 and 3 show advantageous behaviors: Cluster 0 has customers with high payments but low purchases, and Cluster 3 has customers with a high spending rate.

Further clustering based on credit limits and installment purchases indicated that Cluster 0 has a credit limit ranging from 0 to 10,000, with normal limits but rarely makes purchases. Cluster 1 has a low credit limit but still makes purchases ranging from 1,000 to 2,000. The distribution for Cluster 2 is uncertain, whereas Cluster 3 has a normal to high credit limit and high purchases.

In terms of balance and purchases, Cluster 0 customers generally have balances but do not make purchases. Cluster 1 has some customers with low balances and purchasing power. Cluster 2 has few balances and low purchasing power. Cluster 3 customers mostly have balances and high purchasing power, frequently updating their balances.

An examination of transaction distribution found that Cluster 3 had a higher percentage of full payment transactions, while Cluster 2 had more purchase transactions. The scatter plot analysis showed that Cluster 0's purchase transactions range from 0 to 25 with a percentage variation from 0 to 0.8. Cluster 1's purchase transactions fall between 0 and 50 with a percentage around 0 to 1. Cluster 2 exhibited a wider distribution, with purchase transactions ranging from 0 to 350 and a percentage from 0 to 1. Cluster 3 had purchase transactions between 0 and 50 with a percentage distribution from 0.1 to 0.9.

When clustering by purchases and credit limit, Cluster 0 mainly includes customers with credit limits above 3,200 and purchases below 5,000, with a tenure of 12. Cluster 1 has similar characteristics to Cluster 0. Cluster 3 includes customers who make purchases ranging from 300 to 2,000 with credit limits from 1,000 to 15,000, while Cluster 2 includes customers making purchases from 1,500 to above 10,000 with varied credit limits above 6,000. This indicates that most customers can make purchases above 1,500 with varying credit limits.

In conclusion, the overall cluster distribution shows that Cluster 1 has the highest number of customers, followed by Clusters 3, 0, and 2.



