# Sign Language Detector
Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning. Language is expressed via the manual signstream in combination with non-manual elements. Sign languages are full-fledged natural languages with their own grammar and lexicon. This means that sign languages are not universal and they are not mutually intelligible, although there are also striking similarities among sign languages.
<br>
This model has two basic properties to understand :-
                  <pre><b>1.</b> It focuses on Hand actions<br><b>2.</b> Uses Tensorflow model for accurate results</pre>

# Data Source
<b><a href="https://www.kaggle.com/datamunge/sign-language-mnist">Kaggle Sign Language MNIST</a></b>
> <b>Label Column</b><br>
> <b>784 pixel Columns(28x28)</b>

# Libraries
> <b>Numpy</b>: 1.16.4<br>
> <b>Pandas</b>: 0.24.2<br>
> <b>Matplotlib</b>: 3.1.0<br>
> <b>Tensorflow</b>: 1.13.1<br>
> <b>OpenCV</b>: 3.3.1<br>
> <b>Pickle</b>: Any

# Convolutional Model(CNN)
## Convolutional Layers
Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

## Pooling Layer
It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice).

## Fully Connected Layer
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. See the Neural Network section of the notes for more information.

## Model Performance
The model did trained very well , at start it had some variation but after 10 epochs of training it provided good accuracy and loss.
![Model_Performance](https://github.com/Ravi-Maurya/Image_Processing/blob/master/Sign_Language/Images/Model_Performance.png)
<hr>

<img src="https://github.com/Ravi-Maurya/Image_Processing/blob/master/Sign_Language/Images/R.png" width=250 height=250>    <img src="https://github.com/Ravi-Maurya/Image_Processing/blob/master/Sign_Language/Images/V.png" width=250 height=250>    <img src="https://github.com/Ravi-Maurya/Image_Processing/blob/master/Sign_Language/Images/P.png" width=250 height=250>

# Method To Run
> 1. Download the Repository
> 2. Run Sign_model.py

# Method To change Model
> 1. Go to Data.txt and download the training data from link.
> 2. Open TrainModel.ipynb
> 3. Run the Notebook by changing the model architecture.
