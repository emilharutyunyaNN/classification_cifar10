An image recognition task on cifar10 dataset
This project is focusing on deploying metric learning, we create (positive, anchor) paired datapoints and pass them through convolution layer
averagepooling layer, and create distance function between them to make classification of either same class or different in the end based on the distance between the images in a datapoint.
In a nutshell essentially learning a distance between images.
Metric learning is a staple when it comes to recognition tasks. (idea based on the paper "Siamese Neural Networks for One-shot Image Recognition")

Methodology:

  -- Metric learning

    -- Convolution layer

    -- Mean pooling layer

    -- Distance function (l2 - euclidian distance)

    -- Adam optimizer

    -- Sparse Categorical Cross Entropy

Model performace

  Accuracy - above 90%
