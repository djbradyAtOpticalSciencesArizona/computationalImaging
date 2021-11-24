You can test the pre-trained U-net model './Unet2D/netXX.pth' by run infer() in cassi.ipynb.

Please notice that this simple end-to-end model is only for the fixed mask 'H' (loaded from './Unet2D/input.mat'), and similar spectral distributions.

So if you want to train you own model with another 'H', please run './Unet2D/train_unet.py' .

For training, you can reduce batch size if you have limited GPU memory.

Training data: 

Due to the similar spectral distribution requirement, we crop other parts of the 'indian pines' HSI with the same wavelengths and different spacial areas as the training data. 

But because of the self similarity in images, we can not select the region close to the test data (101th pixel on x-axis). So we have to crop the region as far as possible but remain enough data to train. 

As a result of this, we select the 1-80th and 122-145th pixels on x-axis to train.

