# decomp-set-sum

#### Lessons from Shape Dataset
* If there are only a few unique values, model will collapse to avg(sum/n)

* if many datapoints have same sum and same set lenght, irrespective of different items in the set, collapse will occur because sum decomposition is not unique

* All items may have uniuqe values, but if they increase gradually the model creates its own distribution (maintaing high value for high unique eg: 18 -> 34 and 1 -> 6)
    ```
    Original: [41, 16, 38, 32, 19,  7,  9, 41,  1, 36, 17, 38, 37, 22, 25]
    Predicted: [26.8, 11.5, 26.6, 23.5, 14.2,  6.7,  8.7, 26.8, 3.5, 22.5, 11.9, 26.6, 22.1, 15.5, 16.6],
    ```


#### Similarity based additional loss
* Visual similarity can be used, but how. 
    * shape 1 ~ shape 2 but shape 1 color != shape 2 color and so on. 
    * works only when value is assigned as val = f(g(x)) rather than a simple ordering. 
    eg. val = color(shape(x)) rather than index of x in a list

    * uniqueness should be maintained: color(shape(x1)) != color(shape(x2))


* f() is the model. In its simplest form, we have a batch of randomly chosen tuples (x, y).
An oracle indicator function: sim(x,y), where
    * if x ~ y, sim(x,y) = 1
    * if x not ~ y, sim(x,y) = -1 
similarity_loss = min (sim(x,y) * MSE(f(x), f(y)))
Here sim() is not trainable, only f() is trainable.
