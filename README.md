# tictactoe-classifier

a weekend puzzle that I posed to my friends at work: given tiles "X", "O", and a space " ", a board is represented as a string of exactly 9 characters. train a classifier of your choice (such as perceptron for simplicity, but any will do), to classify boards as valid or invalid based on whether you could reach them in a game of tic tac toe.

this repo is my attempt.

the inital work was done with a generator that was not very formal, just randomly populating boards, and anther to generate random valid boards. I ended up with more boards, and many more duplicates than I needed. a formal combinations approach removes the duplicates. this is seen to improve the performance slightly of the base perceptron.

## summary results

a note on deviation from initial proposal:

I started working with strings representing a board, so <code>"X O X "</code> would be a board like this:

<pre>
  <code>X _ _</code>
  <code>O _ _</code>
  <code>X _ _</code>
</pre>

So I had generated csvs with 2 columns: **board**, **label**. However, when I went on to build a tensorflow model, separating out each category as its own column in the csv became important. The schema for the csv of these can be notated with letters and number for the row columns, like this: **a1**, **a2**, **a3**, **b1**, **b2**, **b3**, **c1**, **c2**, **c3**, **label**

testing was done with 0.2 percent of boards in reserve and a variable number of in training.

### perceptron

<small>_tl-dr; ≈60%_</small>

- at .25% of max (39 samples) in training, 58.7% accuracy (4000 runs)
- at 1% of max (157 samples) in training, 59.278% accuracy (1000 runs)
- at 10% of max (1574 samples) in training, 61.398% accuracy (100 runs)
- at 25% of max (3936 samples) in training, 61.027% accuracy (40 runs)
- at 50% of max (7873 samples) in training, 59.747% accuracy (100 runs)
- at 100% of max (15746 samples) in training, 60.35% accuracy (100 runs)

I ran the 25% one at 100 runs also, but got such a different number (55%) that I decided that I can't remove significant variability with less than some number probably related to the 3937 test sample size.

### fully connected neural network

<small>_tl-dr; ≈99.8%_</small>

my keras model was based on this [kdnuggets](https://www.kdnuggets.com/2017/09/neural-networks-tic-tac-toe-keras.html) article. little work was needed to get that to function for our purposes (first attempts already resolved about 94%), so I played with the layers and epochs and learning rate. Although the article highlights an approach to configuring layers and nodes of your network, and using that approach I was able to produce better results (43 epcohs, 99.82% over ≈100000 samples with 20% in test) with 3 • 14 node layers, I am able to achieve nearly this result with a single 9 node layer (17 epochs, 99.2% over sample data). Finally, letting the model run a without early stopping (858 epochs) once on 100000 samples, the accuracy was 99.92%.

While we get good results training on sample sample sizes, we cannot compete with the perceptron for tiny sample sizes (100 samples produce ≈55% accuracy). However, on as little as 333 of-each (≈1000 samples) with the larger network we are approaching our final accuracy (98.5%).

### other methods

the failure for the perceptron to predict made me want to go back and address that. I knew that the problem was that the data is not linearly separaable. So, kernel methods are the solution.

#### support vectors with gaussian kernel

<small>_tl-dr; ≈70%_</small>

this method is an order of magnitude slower, and shows more variance.

each measurement was run 10 times:

- at .25% of max (39 samples) in training, 61.791% accuracy (400 runs) (12KB kernel)
- at 1% of max (157 samples) in training, 63.85% accuracy (400 runs) (197KB)
- at 10% of max (1574 samples) in training, 78.633% accuracy (100 runs) (19.8MB)
- at 25% of max (3936 samples) in training, 78.537% accuracy (100 runs) (124MB)
- at 50% of max (7873 samples) in training, 88.476% accuracy (100 runs) (495MB)
- at 100% of max (15746 samples) in training, 73.505% accuracy (10 runs) (1.98GB kernel)

I am blown away by how well this performs with small sample sizes. relative to the basic perceptron, this is better with sparse samples when working with clean data (no duplicates).

#### direct kernel perceptron

wip

## commands

### generate.py

used to generate data: `/usr/local/opt/python/bin/python3.7 ./generate.py --help` for usage.

### main.py

will be used to actually train and run the perceptron classifier on csv data. accepts paramters, and can be run with the `--help` flag for details.

### keras_classifier.py

can be run when you have generated a file of samples called `samples-tictactoe.csv`. This script must be configured by hand.

## libraries

### classifier.py

a tool to train and test on my data (it will be opinionated about the data)

### perceptron.py

a general 19 line perceptron by Thomas Countz (this is similar but more general and refined to an earlier one I shared with the group)

### generator.py

a collection of methods for generating sample data

### tic_tac_toe.py

helper methods for the generator (it would be illegal to use this class's `is_valid` and `win_check` and other simular methods in the classifier)
