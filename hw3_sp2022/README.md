densenet

The original LR too small and model did not reach peak performance around .867. Increase by order of magnitude. When increased too greatly, fluctuated at too high intervals around this level of performance.

convnet

Larger batch sizes, while able to achieve similar training error as smaller batch sizes, tend to generalize worse to test data, so I kept batch size constant. I used a larger learning rate and more epochs to reach optimal performance around .855. Using a larger batch size allowed me to achieve high accuracy in fewer epochs when keeping learning rate constant, but varied more around the plateau point and did not reach quite the same accuracy.

bestmodel

From previous sections, I learned that CNN is effective for training on image data. I continued to experiment with learning rate, batch size, and epochs after defining a few models that combine convolutional and linear layers. I found that max pooling was useful since the images have high contrast against the background. Furthermore, I found that adding more convolutional layers increased abstraction, while adding some fully connected layers at the end allowed for combining the benefits of dense layers and sparser layers by pooling and flattening data prior to feeding into linear layers. Finally, dropout was added to prevent overfitting with the model.

For the bestmodel, a learning rate of 0.001, batch size of 100, and 2000 epochs were used for optimal performance at the end of training.