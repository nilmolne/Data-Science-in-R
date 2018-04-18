# Data-Science-in-R
In this repository you will find common Data Science challenges I've come across with and solved by using R.

## Create a GAM Formula

Generally, when fitting a Generalized Additive Model (GAM) by using *mgcv*, you need to write a hardcoded formula specifying each predictor parameters' of choice such as the base dimension for penalized regression smoothers (*k*), and the type of penalized smoothing basis (*spline*). This function comes handy to create a formula when all your predictors share same set of parameters (*k*, *spline*).
