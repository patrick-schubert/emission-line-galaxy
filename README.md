# Unsupervised Deep Learning Class Inference with Emission Line Galaxy Data

Project from November - 2018 to June - 2019 and was my very first research project

This repository contains code for the **Unsupervised Deep Learning Class Inference with Emission Line Galaxy Data** Project.

### Summary

This project aimed to propose a principled aproach to classify emission line galaxies. Past methods didn't rely on clear mathmatical fundations and did not enlighted how some classes diverged from another.

All training procedures were taken with high files flow due to training schedueling protocols. Pre-processing analysis and Post-Training statistics/plots were done in a interactive fashion with python notebook sessions.

Models:
- AutoEncoders
- Convolutional AutoEncoders
- Convolutional Variational AutoEncoders
- Variational Autoencoders
- Adversarial Autoencoders

### Training Stats
Loss/Epoch plot for Adversarial AutoEncoder Model

![loss/epoch](https://github.com/patrick-schubert/emission-line-galaxy/blob/main/Misc/AAE_training_loss.png)

### Visualizing Latent Space
Visualization of trained AutoEncoder's bottleneck 

![ls1](https://github.com/patrick-schubert/emission-line-galaxy/blob/main/Plots/CAE/%5BDim%202%5DLatent_Space-Type_whan.png)
<img src="https://github.com/patrick-schubert/emission-line-galaxy/blob/main/Plots/CVAE/%5BDim%204%20_%20STD%200.1%5DLatent_Space-Type_whan.png" width = "786">



### Clustering Statistcs
Bayesian Information Criteria and GAP Statistics for VAE with STD 1

<table border="0">
<tr>
    <td>
    <img src="https://github.com/patrick-schubert/emission-line-galaxy/blob/main/Plots/VAE/%5BDim%204%20_%20STD%201%5DBIC.png" width="100%" />
    </td>
    <td>
    <img src="https://github.com/patrick-schubert/emission-line-galaxy/blob/main/Plots/VAE/%5BDim%204%20_%20STD%201%5DGAP.png", width="100%" />
    </td>
</tr>
</table>


