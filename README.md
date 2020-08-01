# tictactoe-classifier

a weekend puzzle that I posed to my friends at work: given tiles "X", "O", and a space " ", a board is represented as a string of exactly 9 characters. train a classifier of your choice (such as perceptron for simplicity, but any will do), to classify boards as valid or invalid based on whether you could reach them in a game of tic tac toe.

this repo is my attempt.

## summary results

### perceptron

<small>_tl-dr; ≈60%_</small>

in my case I found that I could get about a 60% rate of failure with the perceptron, but no more. Sample sizes ranged from ≈10 to ≈100000. (--of-each 3 to 33333). At about 45 training samples was sufficient to achieve that ratio testing even 10000 test samples.

I was going to move on to a CNN but I realized taht a 3x3 matrix doesnt give much room to shift around in it. additionally, a CNN would not perform better than a fully connected NN here, since a convolutional neural network _zooms in_ on small segments to identify key features of a much larger block of data. A fully connnected neural network instead looks at each location statically, with equal weight.

### fully connected neural network

<small>_tl-dr; >99.1%_</small>

my keras model was based on this [kdnuggets](https://www.kdnuggets.com/2017/09/neural-networks-tic-tac-toe-keras.html) article. little work was needed to get that to function for our purposes, so I played with the layers and epochs and learning rate. Although the article highlights an approach to configuring layers and nodes of your network, and using that approach I was able to produce better results (99.675% over ≈100000 samples with 20% in test) with 3 • 14 node layers, I am able to achieve nearly this result with a single 9 node layer (99.13% over sample data).

While we get good results training on sample sample sizes, we cannot compete with the perceptron for tiny sample sizes (45 samples produce about 11% accuracy). However, on as little as 666 of-each (≈2000 samples) with the larger network we are approaching our final accuracy (98.75% vs 85.75%). The single hidden layer network requires about 10000 samples to produce good predictions.

#### further reading

According to a [summary](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/) by Jason Brownlee, in regards to tuning nodes and layers, one heuristic versus another will perform about as well given a varied enough number of problems to apply them to. It is strange that there is no formal approach to determining such key characteristics.

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
