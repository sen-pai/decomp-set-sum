# decomp-set-sum

#### Lessons from Shape Dataset

* If there are only a few unique values, model will collapse to ``avg over batches (sum of set/n)``

* if many datapoints have same sum and same set lenght, irrespective of different items in the set, collapse will occur because sum decomposition is not unique

* All items may have uniuqe values, but if they increase gradually the model creates its own distribution (maintaing high value for high unique eg: 18 -> 34 and 1 -> 6)

    ```
    MSE = 44
    Original: [41, 16, 38, 32, 19,  7,  9, 41,  1, 36, 17, 38, 37, 22, 25]
    Predicted: [26.8, 11.5, 26.6, 23.5, 14.2,  6.7,  8.7, 26.8, 3.5, 22.5, 11.9, 26.6, 22.1, 15.5, 16.6],
    ```

* Curriculum helped a lot! Initially set len was 2-10 which resulted in okish sum loss ~44 after 50 epochs, reduced set len to 2-5 and sum loss reduced to ~11 MSE after only 30 epochs.
Further because training is on a set, during test time the set size can be any large number (far greater than max during training (10)). 
My guess why this worked is because sum decomposition is not unique. Smaller sets have more descriptive info than a large set.


    ```
    MSE = 18
    Original: [28,  8, 11, 43, 25,  5, 46, 26,  2]
    Predicted: [27.2,  7.4, 10.3, 42.3, 23.7,  4.6, 43.3, 25.4, 1.5]
    ```

    ```
    MSE = 12.2
    Test Set len = 20
    Original: [42, 34, 31, 21, 30, 15, 47, 17, 32, 38, 25, 26,40, 14, 13, 40, 22,  6, 24, 47]
    Predicted: [40.9, 32.3, 30.1, 19.8, 29.3, 14.3, 45.9, 17.1, 30.2, 38.5, 22.8, 26.0, 39.1, 14.4, 13.1, 39.1, 21.9,  6.1, 23.2, 45.9]
    ```

* Even after curriculum it kept neglecting smaller target values. My guess is because even if it gets smaller values (<5) wrong and squashes it to an avg or fixed value like 0 the overall sum based MSE does not get affected as much.
This is surprising because visually there still is a major contrast!

    ```
    MSE = 10.8
    Original: [ 6, 25,  5, 23, 33,  0,  0,  1]
    Predicted: [ 6.5481, 24.8862,  4.0904, 21.8106, 32.3578,  0.0000,  0.0000,  0.0000]
    ```
#### Similarity based additional loss

* Visual similarity can be used, but how.
  * shape 1 ~ shape 2 but shape 1 color != shape 2 color and so on.
  * works only when value is assigned as val = f(g(x)) rather than a simple ordering.
    eg. val = color(shape(x)) rather than index of x in a list

  * uniqueness should be maintained: ``color(shape(x1)) != color(shape(x2))``

* f() is the model. In its simplest form, we have a batch of randomly chosen tuples (x, y).
An oracle indicator function: sim(x,y), where
```
  * if x ~ y, sim(x,y) = 1
  * if x not ~ y, sim(x,y) = -1
```
``similarity_loss = minimize (sim(x,y) * MSE(f(x), f(y)))``
Here sim() is not trainable, only f() is trainable.

I think indirectly I am creating a classification based loss
when sim() -> -1 to 1 rather than -1 or 1, it will be better suited for real world tasks

Implemented in [this script](./train_shapes_vis_oracle.py). Acc just shot up! 
```
Epoch = 2
Sum MSE = 70
Vis MSE = -590
Original: [37, 44,  0, 11, 28, 42, 16, 28, 45, 16, 28, 34, 24, 14, 17]
Predicted: [39.0, 49.8,  0.2,  8.1, 31.1, 53.2, 12.1, 31.1, 47.4, 12.1, 31.1, 34.5, 23.8, 10.5, 10.6]
```
There is a very clear sim(x,y) = -1 imbalance over 1, 99% of the time we can expect -1. 
Maybe MAE is a better loss rather than MSE for this?. 

```
MAE + 50% balance 
Epoch = 5
Sum MSE = 24
Vis MAE = -1.18
Original: [28, 17,  7,  9, 16, 15,  4,  9, 15,  2, 18, 21, 21, 47,  6, 14]
Predicted: [28.5, 16.9,  7.0,  9.2, 16.7, 15.2,  3.3,  9.2, 15.2,  2.5, 18.4, 21.8, 21.8, 48.8,  6.1, 14.5]
```


#### Free lunch?

Because we have ``X = [x1 ,x2 .. xn]`` and we know ``y = sum of f(xi)``, can we predict a set ``Y = [f(x1), f(x2).. f(x_{n-1})]`` and then infer ``f(xn) = y - Sum of Y`` ?

Maybe a loss based on this idea? but how would be know item based ``f(xn)`` true value? 

Rather we play with sets? By reframing ``X = A + B`` and ``y = sum of f(A) + sum of f(B)``
We predict ``f(a_i)`` and create a loss based on
``MSE(sum of f(B), (y - sum of true f(A)))`` 



#### Small Farms are better

* Small set lenght
* small yield values 


#### Main Questions:

* Are there other tasks in remote sensing that preserve sum composition. 
* Any datasets which are publically available for yield administrative zones?




#### Vegetation Dryness

##### Set def
* closest points
* random from train

##### Test splits:
* based on Area
* based on Time

#### Results

* Train = 2015, 2016, 2017 and Test = 2018, 2019
  ```
  With Loss as MSE
  Original: [132.0, 109.6,  56.0,  92.5, 214.4,  76.6, 109.0, 92.5, 287.0, 165.0,  85.0, 104.0,  92.5,  90.0, 104.5,  85.8,  85.8]
  Predicted: [113.0, 103.4, 104.3, 108.2, 197.6, 101.0, 107.5, 115.0, 112.6,  96.2, 100.3,  92.0,  83.4,  84.5,107.3, 104.2, 104.2])
  ```

  ```
  With Loss as L1
  Original: [108.0,  89.5, 110.6, 122.0,  63.2,  81.0]
  Predicted: [113.2,  96.5, 107.3,  86.3,  77.2,  74.8]
  ```

* Train Test Split based on Site.
  ```
  RMSE = 23.07
  Original: [ 99.6, 102.0,  63.0,  52.5,  72.6, 143.1, 108.0, 84.6,  99.3,  99.0, 101.3]
  Predicted: [ 91.2, 102.1,  90.2,  81.9,  96.9, 111.2,  79.6, 102.3,  89.5, 100.0,  83.8]
  ```

  [Original Paper](https://www.sciencedirect.com/science/article/pii/S003442572030167X) has RMSE = 25. 
  We achieve RMSE of 21 with a simple 2 layer MLP, no normalization, no site info, no time info.
  Original paper had a target **per** pixel, we just have the total sum

Model is bad at the extreme points, as expected (Maybe Small Sets can help here ). 