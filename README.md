# traffic-sign-classifier

Experimentation Process

For my quick experimentation process, I tried 34 for the number of filters for the Conv2D layer, with a 3x3, (2, 2) pool size for the pooling layers, a flatten layer, 2 hidden layers: one with 128 neurons (units) and 84 neurons, and used the adam optimizer borrowing some ideas from the lecture code. This worked well for me because the compilation time was not too long (approx 17 minutes). My first run yeilded the following results: accuracy: 0.7264 - loss: 0.9340.

```
Epoch 1/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 22ms/step - accuracy: 0.1959 - loss: 4.3541
Epoch 2/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.4232 - loss: 2.0953 
Epoch 3/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.5310 - loss: 1.5900 
Epoch 4/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 21ms/step - accuracy: 0.5864 - loss: 1.3368 
Epoch 5/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.6356 - loss: 1.1554 
Epoch 6/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 20s 22ms/step - accuracy: 0.6680 - loss: 1.0316
Epoch 7/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.6812 - loss: 1.0116 
Epoch 8/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 21ms/step - accuracy: 0.7088 - loss: 0.9133 
Epoch 9/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 11s 21ms/step - accuracy: 0.7282 - loss: 0.8405 
Epoch 10/10 500/500 ━━━━━━━━━━━━━━━━━━━━ 12s 23ms/step - accuracy: 0.7342 - loss: 0.8098 333/333 - 2s - 6ms/step - accuracy: 0.7264 - loss: 0.9340
```

What worked well was increasing the number of filters and kernel size (Conv2D: 34 filters, 3×3 → 128 filters, 5×5). I noticed too that increasing the neurons for my hidden layers (Dense: 128 → 256, Dense: 84 → 128) worked even better. The run time seemed shorter too though my first run's duration was a rought estimate. On my second run, I reach 91.6% accuracy and hit these results:

Training accuracy: 93.89% Training loss: 0.2463 Test accuracy: 91.59% Test evaluation time: 11 min 23 seconds
