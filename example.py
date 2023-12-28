import json

from pandas import read_csv

from SCDM import SymbolicCDM

print("Load dataset config")
# change the path to load different datasets
with open("dataset/FracSub/config.json") as configFile:
    config = json.load(configFile)
print("Load dataset: ", config["dataset"])
print("Load Q matrix from: ", config["qMatrixPath"])
q_matrix = read_csv(config["qMatrixPath"], header=None).to_numpy(dtype=int)
print("Load response logs from: ", config["dataPath"])
response_logs = read_csv(config["dataPath"], header=None).to_numpy(dtype=int)
print("Load successfully! Record: %d" % (len(response_logs)))
print("Create cognitive diagnosis model")

model = SymbolicCDM(q_matrix,
                    config["studentNumber"], config["questionNumber"], config["knowledgeNumber"],
                    response_logs)
print("Create Successfully")

model.train(epochs=3, nn_epochs=100)
