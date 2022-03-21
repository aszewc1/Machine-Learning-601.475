densenet.py

The original LR too small and model did not reach peak performance around .867. Increase by order of magnitude. When increased too greatly, fluctuated at too high intervals around this level of performance.

Unsure about 15.8.b

Larger batch sizes, while able to achieve similar training error as smaller batch sizes, tend to generalize worse to test data, so I kept batch size constant. I used a larger learning rate and more epochs to reach optimal performance around .855. Using a larger batch size allowed me to achieve high accuracy in fewer epochs when keeping learning rate constant, but varied more around the plateau point and did not reach quite the same accuracy.