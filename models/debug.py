import pickle as pkl

import pdb 

with open("/home/estengel/scratch/nan_loss_batch.pkl", "rb") as inputf, open("/home/estengel/scratch/nan_loss_batch_outputs.pkl", "rb") as outputf:
    batch = pkl.load(inputf)
    output = pkl.load(outputf) 

pdb.set_trace() 
