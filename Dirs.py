import os
models = ["AE", "CAE", "VAE", "CVAE", "AAE"]

root_dir = os.getcwd()

models_dir = root_dir + "/Models"

plots_dir = root_dir + "/Plots"

lossEpoch_dir = root_dir + "/Loss_Epoch" 

latentSpace_dir = root_dir + "/Latent_Space"

tsne_dir = root_dir + "/T-SNE"

tsneScore_dir = root_dir + "/T-SNE Cluster Scores"

ourclusters_dir = root_dir + "/OurClusters"

arielData_dir = root_dir + "/ArielData"

for i in range(len(models)):
    if not os.path.isdir(plots_dir + "/" + models[i]):
        os.mkdir(plots_dir + "/" + models[i])


    