import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from Dirs import *

print("Iniciando...")
start = time.time()
arquivos = os.listdir(lossEpoch_dir)
for arquivo in range(len(arquivos)):
    print(arquivos[arquivo])
    df = pd.read_csv(lossEpoch_dir + '/' + arquivos[arquivo])
    fig = plt.figure(figsize=(20, 5))
    for coluna in df:
        ax = fig.add_subplot(111)
        ax.plot(df[coluna], label = coluna)

    plt.legend(loc = "upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    for modelo in range(len(models)):
        if models[modelo] in arquivos[arquivo]:
            fragmentos_arquivo = arquivos[arquivo].split("||")
            if len(fragmentos_arquivo) == 2:
                dim = fragmentos_arquivo[1].split("=")
                plt.savefig(plots_dir + "/" + models[modelo] + "/[{} {}]Loss_Epoch.png".format(dim[0], dim[1]))
            else:
                dim = fragmentos_arquivo[1].split("=")
                std = fragmentos_arquivo[2].split("=")
                plt.savefig(plots_dir + "/" + models[modelo] + "/[{} {} | {} {}]Loss_Epoch.png".format(dim[0], dim[1], std[0], std[1]))
                
finish = time.time()
tempo = finish - start
print("Minutos: {}".format(int(tempo/60)))