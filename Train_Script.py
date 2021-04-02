#from Lat_Gen import Lat_Gen
def setup(mode="", model="", dim=0, std=0, epochs=10):
    
    modes = ["train", "plot"]
    models = ["AE", "CAE", "VAE", "CVAE", "AAE"]

    if mode not in modes:
        print("Define a mode: train or plot")
    elif model not in models:
        print ("Define a model: AE, CAE, VAE or CVAE")
    elif model == "VAE" and std == float("0"):
        print("Define a std value")
    elif dim <= 0:
        print("Define a positive latent dimension")
    else:
        
        if mode == "train":
            from Train import train
            train(model,std,dim,epochs)
            
        else:
            if mode == "plot":
                from Plot import plot
                plot(model,std,dim)




print("****Iniciando****")
model_list = ["CVAE"]
std_list = [1.0,0.5, 0.1 ,1e-13]
lat_dim_list = [2,3,4,5,6]

for i in range(len(model_list)):
    for j in range(len(std_list)):
        for k in range(len(lat_dim_list)):
            model = model_list[i]
            lat_dim = lat_dim_list[k]
            std = std_list[j]
            if (model == "CAE" or model == "AE") and j != 0:
                continue
            else:
                
                setup(mode="train", model= model, dim=lat_dim,std=std, epochs=50)



print("****FIM****")
#Lat_Gen()
