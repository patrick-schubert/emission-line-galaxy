import pandas as pd
import numpy as np


def train(model, std, dim,epochs):
    
    batch_size = 200
    
    if model == "VAE":
        from Model import VAE
        model = VAE
        for i in range(10):
            print("")  
        print("Training VAE")
        for i in range(10):
            print("")
        model.fit(dim, std, epochs, batch_size)
    elif model == "CVAE":
        from Model import CVAE
        model = CVAE
        for i in range(10):
            print("")  
        print("Training CVAE")
        for i in range(10):
            print("")
        model.fit(dim, std, epochs, batch_size)
    elif model == "AE":
        from Model import AE
        model = AE 
        print(f"Training AE | Lat_Dim={dim}")
        model.fit(dim, std, epochs, batch_size)
    elif model == "CAE":
        from Model import CAE
        model = CAE
        for i in range(10):
            print("")  
        print(f"***Training CAE | Dim={dim}***")
        for i in range(10):
            print("")
        model.fit(dim, std, epochs, batch_size)
    else:
        from Model import AAE
        model = AAE
        for i in range(10):
            print("")
        print("Training AAE | Lat_Dim={} | STD={}".format(dim, std))
        for i in range(10):
            print("")
        stats = 0
        while(stats == 0):
            stats = model.build(dim, std, "train")