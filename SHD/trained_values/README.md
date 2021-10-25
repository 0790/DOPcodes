The following folder contains the trained weights for the following networks:
1. SNN with one layer and 128 units, trained with lr = 0.00005 and uniform distribution for weight initialization.
2. SNN with one layer and 128 units, trained with lr = 0.00005 and normal distribution for weight initialization.
3. SNN with one layer and 256 units, trained with lr = 0.00005

4. SNN with two layers and 128 units, trained with lr = 0.00005
5. SNN with two layers and 256 units, trained with lr = 0.00005

6. RSNN with 128 units, trained with lr = 0.00005
6. RSNN with 256 units, trained with lr = 0.00005

These are present in the .pt files
Using torch.load to save values between run changes.

.pkl files are used to store the loss value for every epoch.
