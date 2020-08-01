# tictactoe-classifier

a weekend puzzle that I posed to my friends at work: given tiles "X", "O", and a space " ", a board is represented as a string of exactly 9 characters. train a classifier of your choice (such as perceptron for simplicity, but any will do), to classify boards as valid or invalid based on whether you could reach them in a game of tic tac toe.

this repo is my attempt.

## commands

### main.py

will be used to actually train and run the classifier on data

### generate.py

used to generate data: `/usr/local/opt/python/bin/python3.7 ./generate.py --help` for usage. I will be trying an intial training run with:

```
/usr/local/opt/python/bin/python3.7 ./generate.py --of-each 3333 --with-labels > samples-tictactoe.csv
```

## libraries

### classifier.py

a tool to train and test on my data (it will be opinionated about the data)

### perceptron.py

a general 19 line perceptron by Thomas Countz (this is similar but more general and refined to an earlier one I shared with the group)

### generator.py

a collection of methods for generating sample data

### tic_tac_toe.py

helper methods for the generator (it would be illegal to use this class's `is_valid` and `win_check` and other simular methods in the classifier)
