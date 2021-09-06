# Contrast_enhanced_GAN

## Contrast_enhanced Image GAN Models

### models.Generator.CycleGAN_G.py : CycleGAN Generator Model Architecture.
### models.Discriminator.CycleGAN_D.py : CycleGAN Dismcriminator Model Architecture.

### train.train_CycleGAN.py : CycleGAN Train py file.
### batch_size = 32 : allocated GPU memory about 77.316GB ( if you use 4 GPU slot, each gpu used memory about 19.329GB )

### Default Hyperparams
#### mini batch size : 32
#### number of total epochs : 200
#### number of decay epochs : 100 ( Linear learning rate decay )
#### learning rate = 0.0002
#### input image size = 256 %2^8%
#### Transforms compose : Horizon flip, resize, normalize std = 0.5, mean = 0.5 (Gray Scale)
#### Optimizer = Adam
#### GAN Loss = MSE
#### Identity Loss = L1 Loss
#### Cycle Loss = L1 Loss



