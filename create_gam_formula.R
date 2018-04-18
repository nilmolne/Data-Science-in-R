

## Example: 
## The following line will create a formula (including the target variable) for
## all predictors fit with 5th dimensional gaussian processes smoothing splines.

## gam.formula <- 
##   create_gam_formula(data['target.variable'], data[!colnames(data) %in% 'target.variable'], 5, 'gp')


create_gam_formula <- 
  function(dep.variable, features, k, gspline) {
    
    dep.var.name <- colnames(dep.variable)
    feature.names <- colnames(features)
    
    smooth.features <- paste0('s(', feature.names, ', k = ', k, ', bs = "', gspline, '")')
    formula <- as.formula(paste0( dep.var.name, ' ~ ', paste0(smooth.features, collapse = ' + ')))
    
    return ( formula )
    
  }
