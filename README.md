# decomp-set-sum

#### Lessons from Shape Dataset
* If there are only a few unique values, model will collapse to avg(sum/n)

* if many datapoints have same sum and same set lenght, irrespective of different items in the set, collapse will occur

* All items may have uniuqe values, but if they increase gradually the model creates its own distribution (maintaing high value for high unique eg: 18 -> 34 and 1 -> 6)


#### Similarity based loss
* Visual similarity can be used, but how. 
    * shape 1 ~ shape 2 but shape 1 color != shape 2 color and so on. 
    * works only when value is assigned as val = f(g(x)) rather than a simple ordering. 
    eg. val = color(shape(x)) rather than index of x in a list

    * uniqueness should be maintained: color(shape(x1)) != color(shape(x2))