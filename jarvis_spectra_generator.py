import pandas as pd
import numpy as np
import csv
from atomgpt.inverse_models.utils import smooth_xrd
from sklearn.model_selection import train_test_split
from jarvis.core.atoms import Atoms
import sys
from jarvis.db.figshare import data
import matplotlib.pyplot as plt


##### For training. ##############################
d = data('dft_3d') #choose a name of dataset from above
df = pd.DataFrame(d)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['crys'],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,  # half of 20% -> 10%
    stratify=temp_df['crys'],
    random_state=42,
)

temp_df, small_test_df =  train_test_split(
    test_df,
    test_size=0.1,  # ~750 points
    stratify=test_df['crys'],
    random_state=42,
)

with open("id_prop_train.csv","w", newline="") as f, open("id_prop_val.csv","w", newline="") as g, open("id_prop_test.csv","w", newline="") as h:
    writer_train = csv.writer(f)
    writer_val   = csv.writer(g)
    writer_test  = csv.writer(h)
    for i in range(len(train_df)):
        # Visualize an atoms object
        atoms = Atoms.from_dict(train_df.iloc[i]['atoms'])
        jid = train_df.iloc[i]['jid']
        intensity_string, intensity = smooth_xrd(atoms)
        row = [f'{jid}'] + [f"{v:.2f}" for v in intensity] #[intensity_string]
        writer_train.writerow(row)
    for i in range(len(val_df)):
        atoms = Atoms.from_dict(val_df.iloc[i]['atoms'])
        jid = val_df.iloc[i]['jid']
        intensity_string, intensity = smooth_xrd(atoms)
        row = [f'{jid}'] + [f"{v:.2f}" for v in intensity]
        writer_val.writerow(row)
    for i in range(len(test_df)):
        atoms = Atoms.from_dict(test_df.iloc[i]['atoms'])
        jid = test_df.iloc[i]['jid']
        intensity_string, intensity = smooth_xrd(atoms)
        row = [f'{jid}'] + [f"{v:.2f}" for v in intensity]
        writer_test.writerow(row)
#####################################################################################################################

##### For evaluation. use small_test_df above ###############
import pandas as pd
import csv
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
import math
from atomgpt.inverse_models.utils import smooth_xrd
import numpy as np
from jarvis.core.atoms import Atoms
import sys
from jarvis.db.figshare import data
import matplotlib.pyplot as plt


crys2num = {
    "triclinic": 1,
    "monoclinic": 2,
    "orthorhombic": 3,
    "tetragonal": 4,
    "trigonal": 5,   # sometimes JARVIS uses "trigonal" instead of "rhombohedral"
    "rhombohedral":5,
    "hexagonal": 6,
    "cubic": 7,
}


d = data('dft_3d') #choose a name of dataset from above
df = pd.DataFrame(d)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['crys'],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,  # half of 20% -> 10%
    stratify=temp_df['crys'],
    random_state=42,
)

temp_df, small_test_df =  train_test_split(
    test_df,
    test_size=0.01,  # ~75 points. adjust smaller first
    stratify=test_df['crys'],
    random_state=42,
)

#### load out_inv.json file ####
out = json.load('out_inv.json')
inn = []
outt = []
systems = []
for i in range(len(small_test_df)):
    atoms = Atoms.from_dict(small_test_df.iloc[i]['atoms'])

    crys_system = crys2num[small_test_df.iloc[i]['crys']]
    a,b,c,v = atoms.lattice.abc, atoms.volume
    predicted_atoms = out[0][i]['atoms']
    at,bt,ct,vt = predicted_atoms.lattice.abc, predicted_atoms.volume
    #_,intensity = smooth_xrd(atoms)
    #_,intensity_pred = smooth_xrd(predicted_atoms)

inn = np.vstack(inn)
outt = np.vstack(outt)
labels = ['a','b','c','V']
colorlist = ['blue', 'green', 'red', 'cyan', 'magenta', 'purple','black']
colors = [colorlist[i] for i in crys_system]
for i in inn.shape[1]:
    plt.scatter(inn[:,i],outt[:,i],c=colors)
    plt.xlabel('Target %s'%labels[i])
    plt.ylabel('Predicted %s'%labels[i])
    if i < inn.shape[1]-1:
        plt.xlim(0,20)
        plt.ylim(0,20)
    else:
        plt.xlim(0,1000)
        plt.ylim(0,1000)
    MAE = np.sum(np.abs(inn[:,i]-outt[:,i]))/len(inn[:,i])
    
#####################################################################
