
In this short project, I used LSTM cells with Tensorflow to generate news headlines with the length of 20.

There are four models, each with different embedding size: 250, 500, 750 and 1000.

The whole training was done on Google Colab and took about an hour to train them all. Because I cannot keep running an instance on Colab due to timeout, I could not try many different combinations of hyperparameters.

The losses of each model is the following.

![Losses](plots/losses.png)

Given a phrase 'samsung and apple', it generated
1. samsung and apple are said in talks with samsung over tablet sales in china - report says no iphone'
2. samsung and apple end patent fight with u.s. smartphone market war with u.s . . . ) says on
3. samsung and apple are reportedly a new smartphone in the u.s . . . . . . it's great
4. samsung and apple ' top pick ' for ipad turns too late ' : ' edition isn't not a
Since they were not trained fully, it does not show any satisfactory results but still be able to demonstrate and show how a model can be built from scratch and trained.
