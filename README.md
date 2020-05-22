# DeepLearning - Project #1 - Compare two Handwritten digits

## Libraries used: 

<ul>
<li>Pytorch</li>
<li>Matplotlib</li>
</ul>


## File description:

<ul>
<li>models.py: Definition of four different models</li>
<li>main.ipynb: Train function and results visualization</li>
<li>dlc_practical_prologue.py: Functions to import dataset properly</li>
</ul>


## Models description:

### First model used for the project, the simplest one.

This first model uses no weight sharing, the two input images are passed through two separate convolutional networks. The output of these 2 networks are then concatenated and passed through 4 fully connected layers. There is a single output neuron indicating if the first digit is larger (output should be 1) or if the second one is (output should be 0).

### Second model, introduces weight sharing for the convolutional layers.

This second model uses weight sharing, the two images are passed through a single convolutional network. The outputs are then concatenated and passed through 4 fully connected layers. There is a single output neuron indicating if the first digit is larger (output should be 1) or if the second one is (output should be 0).

### Third model, we use the label of the digits as an auxiliary loss.

This third model introduces a set of new outputs. The network must now also predict the digit of each image. This is used as an auxiliary loss. The two images are passed through a single convolutional network. The outputs are concatenated and passed through a series of fully connected layers. The network then has 3 outputs: 
<ul>
<li>A single neuron to predict which digit is the largest.</li>
<li>Two softmax outputs two predict the labels of the 2 images. These two outputs use weight sharing.</li>
</ul>

### Fourth model, we also use the label of the digits as an auxiliary loss.

This fourth model adds batch normalisation on top of the third model.
The network then has 3 outputs:
<ul>
<li>A single neuron to predict which digit is the largest.</li>
<li>Two softmax outputs two predict the labels of the 2 images. These two outputs use weight sharing.</li>
</ul>


## Execution:

To run the program how it is intended you need to execute test.py.
You will first be asked if you want to train the models over one round or multiple rounds. Just make your choice by inputing a 0 or 1 and pressing ENTER. You can also just input the ENTER key every time you're asked a choice to choose the default value. You will be then be asked how many epochs you want to train over. If you chose multiple rounds, you will be asked to choose a number of rounds and it will also ask you at the end if you want to plot the results.
