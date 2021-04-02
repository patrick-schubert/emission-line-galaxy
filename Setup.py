import argparse

parser = argparse.ArgumentParser()


def setup():


    parser.add_argument("--mode", default="")
    parser.add_argument("--model", default = "")
    parser.add_argument("--std", default="0")
    parser.add_argument("--lat_dim", default="0")
    parser.add_argument("--epochs", default="10")
    args = parser.parse_args()

    
    mode = args.mode
    model = args.model
    std = float(args.std)
    dim = int(args.lat_dim)
    epochs = int(args.epochs)
    
    modes = ["train", "plot"]
    models = ["AE", "CAE", "VAE", "CVAE"]

    if mode not in modes:
        print("Define a mode: train, lat_gen or plot")
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
        elif mode == "lat_gen":
            from Lat_Gen import lat_gen
            lat_gen(model,std,dim)
        else:
            if mode == "plot":
                from Plot import plot
                plot(model,std,dim)
setup()