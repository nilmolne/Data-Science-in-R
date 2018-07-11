# Data-Science-in-R
In this repository you will find common Data Science challenges I've come across with and solved by using R.

## Predict Time Until Next Order

See [Time_Until_Next_Order.Rmd]()

## Create a GAM Formula

Generally, when fitting a Generalized Additive Model (GAM) by using *mgcv*, you need to write a hardcoded formula specifying each chosen predictor parameters' such as the base dimension for penalized regression smoothers (*k*), and the type of penalized smoothing basis (*spline*). This function comes handy to create a formula when all your predictors share the same set of parameters (*k*, *spline*).

See [create_gam_formula.R](https://github.com/nilmolne/Data-Science-in-R/blob/master/create_gam_formula.R)
