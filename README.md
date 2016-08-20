# Tensorflow Classifiers

The goal of this project is to present various types of neural networks in a manner where if you understand one, that it would be easy to transition to another.

So, simply choose a branch you're interested/comfortable in and do `git diff your_branch..other_branch` study the deltas : )

Aiming for the `master` branch to have the (subjectively) coolest of our neural network demos.

## Where to begin
Start at the `cat-dog-classifier-fc-ffnn` has a more vanilla classifier and test files, and wouled recommend starting there, adding more images adjusting the parameters, then diffing forward.

see the git commands below for some useful commands

## Necessary git commands


### to get a particular branch to your local machine

`git pull origin <branch_name>`

example:

`git pull origin cat-dog-classifier-fc-ffnn`

### to compare branches just type:

`git diff <branch_one>..<branch_two`

example:

`git diff master..cat-dog-classifier-fc-ffnn`

### to move to a different branch (once downloaded on your local machine)

`git checkout <branch_name>`

example:

`git checkout cat-dog-classifier-fc-ffnn`


## Contributing

Feel free to contribute, branch naming conventions below:

`<description-of-branch>-<type-of-network><github-username>`

example:

if we have a cat and dog classifier, fully-connected feed-forward neural net, and my username is gskielian, then the branch name would be:

`cat-dog-classifier-fc-ffnn-gskielian`

thanks!
