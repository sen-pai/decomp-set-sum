# decomp-set-sum

#### Lessons from Shape Dataset
* If there are only a few unique values, model will collapse to avg(sum/n)

* if many datapoints have same sum and same set lenght, irrespective of different items in the set, collapse will occur

* All items may have uniuqe values, but if they increase gradually the model creates its own distribution (maintaing high value for high unique eg: 18 -> 34 and 1 -> 6)