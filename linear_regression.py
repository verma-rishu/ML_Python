from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)
'''
    hm = how many datapoint to create
    variance = variability of the dataset
    step = how far from avg is the data point
    correlation = +ve, -ve, none (True = +ve, False = -ve)
'''
def create_dataset(hm, variance, step=2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs))**2 - mean(xs**2))
        )
    b = mean(ys) - (m * mean(xs))
    return m,b

def squared_error(ys_orig, ys_regression_line):
    return sum((ys_regression_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_regression_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_regression_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return (1- (squared_error_regr/squared_error_mean))

'''
    Current vaiance = 40 : COD = 0.58
    Decrease variance = 10, coefficent of determination increases (gets better) = 0.92
    Increase variance = 80, coefficent of determination decreses = 0.43
'''
xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

#how good of a fit is our best fit line
'''
    R_Squared above 0 means the regression line was more accurate but you need 
    to determine your own coefficeint of determination based on the scenario.
'''
r_squared = coefficient_of_determination(ys, regression_line)
print(f'R squared = {r_squared}')

'''
    Predicting for any given x
'''
predict_x = 8 
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, s= 100, color='g')
plt.plot(xs, regression_line)
plt.show()