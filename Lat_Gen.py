from Dirs import *
import Layers
import Model
from Data import *
import pandas as pd
import numpy as np
import time


def Lat_Gen():
    print("Iniciando...")
    start = time.time()



    model_list = ["CAE","CVAE"]
    #std_list = [1.0,0.5, 0.1, 1e-13]
    std_list = [1.0,0.5, 0.1, 1e-13]
    lat_dim_list = [2,3,4,5,6]

    for i in range(len(model_list)):
        for j in range(len(std_list)):
            for k in range(len(lat_dim_list)):
                
                model = model_list[i]
                lat_dim = lat_dim_list[k]
                std = std_list[j]
                
                if (model == "AE" or model == "CAE") and j != 0:
                    continue
                    
                else:

                    #print(f"[CAE]Lat_Dim={lat_dim}")

                    #model = Model.VAE.build(dim=lat_dim, std=std)
                    #model.load_weights(models_dir + "/" + f"[VAE]Lat_Dim={lat_dim} | STD={std}.hdf5")
                    #modelo = Layers.VAE.build(dim=lat_dim ,std= std)

                    #Model.AAE.build(dim = lat_dim, std= std, mode="Test")
                    #predict = modelo.predict(data_normalized)
                    #df = pd.DataFrame(data = predict)
                    #df.to_csv(latentSpace_dir + "/" + f"[AAE]Lat_Dim={lat_dim} | STD={std}.csv", index=False)
                    #break
                    if model == "CVAE":
                        print(f"[CVAE]Lat_Dim={lat_dim} | STD={std}")
                        model = Model.CVAE.build(dim=lat_dim, std=std)
                        model.load_weights(models_dir + "/" + f"[CVAE]Lat_Dim={lat_dim} | STD={std}.hdf5")
                        modelo = Layers.CVAE.build(dim=lat_dim ,std= std)
                        modelo.set_weights(model.get_weights())
                        predict = modelo.predict(cx_test, batch_size = 16)
                        df = pd.DataFrame(data = predict)
                        df.to_csv(latentSpace_dir + "/" + f"[CVAE]Lat_Dim={lat_dim} | STD={std}", index=False)

                    
                    if model == "VAE":
                        print(f"[VAE]Lat_Dim={lat_dim} | STD={std}")
                        model = Model.VAE.build(dim=lat_dim, std=std)
                        model.load_weights(models_dir + "/" + f"[VAE]Lat_Dim={lat_dim} | STD={std}.hdf5")
                        modelo = Layers.VAE.build(dim=lat_dim ,std= std)
                        modelo.set_weights(model.get_weights())
                        predict = modelo.predict(x_test, batch_size = 16)
                        df = pd.DataFrame(data = predict)
                        df.to_csv(latentSpace_dir + "/" + f"[VAE]Lat_Dim={lat_dim} | STD={std}", index=False)

                    
                    if model == "AE":
                        print(f"[AE]Lat_Dim={lat_dim}")
                        model = Model.AE.build(dim=lat_dim)
                        model.load_weights(models_dir + "/" + f"[AE]Lat_Dim={lat_dim}.hdf5")
                        modelo = Layers.AE.build(dim=lat_dim)
                        modelo.set_weights(model.get_weights())
                        predict = modelo.predict(x_test, batch_size = 16)
                        #print(modelo.summary())
                        df = pd.DataFrame(data = predict)
                        df.to_csv(latentSpace_dir + "/" + f"[AE]Lat_Dim={lat_dim}", index=False)
                    
                    if model == "CAE":
                        print(f"[CAE]Lat_Dim={lat_dim}")
                        model = Model.CAE.build(dim=lat_dim)
                        #print(model.get_weights()[0][0])
                        model.load_weights(models_dir + "/" + f"[CAE]Lat_Dim={lat_dim}.hdf5")
                        #print(model.get_weights()[0][0])
                        modelo = Layers.CAE.build(dim=lat_dim)
                        #print(modelo.get_weights()[0][0])
                        modelo.set_weights(model.get_weights())
                        #print(modelo.get_weights()[0][0])
                        predict = modelo.predict(cx_test, batch_size = 200)
                        #print(modelo.summary())
                        df = pd.DataFrame(data = predict)
                        df.to_csv(latentSpace_dir + "/" + f"[CAE]Lat_Dim={lat_dim}", index=False)


    finish = time.time()
    tempo = finish - start
    print("Minutos: {}".format(int(tempo/60)))
Lat_Gen()
