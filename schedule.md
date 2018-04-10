# Recommendation System

## Average Fillling

###Problem Definition

**Input : ** $R={(u,i,r_{ui})}$  observed rating records
**Output: **  for each record $(u,i,r_{ui})$ in $R^{te}$, estimate the rating $\hat {r}_{ui}$
**Evaluation: ** how accurate is the predicted rating $\hat{r}_{ui}$

###Compute parameters
1. global average : $\overline{r} = \Sigma  r/ |r|$   where $r$ is the nonzero rating vector 
2. average rating of user $u$ :  $\overline{r}_u = \Sigma  r_u / |r_u|$  where $r_u$ is the nonzero rating  vector of user $u$
3. average rating of item $i$ :  $\overline{r}_i  = \Sigma  r_i / |r_i| $ where $r_i$ is the nonzero rating vector of user $i$
4. bias of user $u$ : $b_u = \Sigma  (r_u - \overline{r}) / |r_u - \overline{r}|$   suppose $i$ is the  vector that user $u$ rated item , $r_u$ is the rating on $i$ vector , $\overline{r}$ is the average on item $i$ vector 
5. bias of item $i$ : $b_i = \Sigma  (r_i  - \overline{r} ) / |(r_i - \overline{r})|$  similiar to $b_u$

### Evaluation Metrices 
1. Mean Absolute Error (MAE)
$$
MAE = \Sigma_{(u,i,r_{ui})\in R^{te}} |r_{ui} - \hat{r}_{ui}| / |R^{te}|
$$

2. Root Mean Square Error (RMSE)
$$
RMSE = \sqrt{\Sigma_{(u,i,r_{ui})\in R^{te}} (r_{ui}- \hat{r}_{ui})^2 / |R^{te}|}
$$

### Result And Prediction Rule

![Figure_1](C:\Users\Jack\OneDrive\图片\屏幕快照\Figure_1.png)

Code is here : 
[average_filling.py][1]  
[1]:  https://github.com/Jack-An/RecommendationSystem/blob/master/algorithm/average_filling.py/



