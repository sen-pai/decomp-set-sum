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
Further because training is on a set, during test time the set size can be any large number (far greater than max during training (10))


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
