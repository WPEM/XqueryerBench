import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ase.db
import json
from scipy.interpolate import interp1d
import ase
import random
import os
class ASEDataset(Dataset):
    def __init__(self, db_paths,encode_element,train=False):
        with open('./CGCNN_atom_emb.json' , 'r') as file:
            self.cgcnn_emb = json.load(file)
        self.db_paths = db_paths
        self.train=train
        self.encode_element = encode_element
        self.dbs = [ase.db.connect(db_path) for db_path in db_paths]
        print("Loaded data from:", db_paths)

    def __len__(self):
        total_length = sum(len(db) for db in self.dbs)
        return total_length

    def __getitem__(self, idx):
        
        cumulative_length = 0
        for i, db in enumerate(self.dbs):
            if idx < cumulative_length + len(db):
                # Adjust the index to the range of the current database
                adjusted_idx = idx - cumulative_length
                row = db.get(adjusted_idx + 1)  # ASE db indexing starts from 1
                if self.encode_element:
                    atoms = row.toatoms()
                    element = self.random_remove_elements(set(atoms.get_chemical_symbols()))
                    element_encode = self.symbol_to_atomic_number(element)
                    element_value = []
                    for code in element_encode:
                        value = self.cgcnn_emb[str(code)]
                        element_value.append(value)
                    # mean pooling
                    element_value=torch.mean(torch.tensor(element_value, dtype=torch.float32),dim=0)
                # Extract relevant data from the row
                #latt_dis = eval(getattr(row, 'latt_dis'))
                if self.train:
                    intensity = self.mixture( eval(getattr(row, 'intensity')) )
                else:
                    intensity = eval(getattr(row, 'intensity')) 
                id_num = getattr(row, 'Label')
                
                # Convert to tensors
                #tensor_latt_dis = torch.tensor(latt_dis, dtype=torch.float32)
                tensor_intensity = torch.tensor(intensity, dtype=torch.float32)
                tensor_id = torch.tensor(id_num, dtype=torch.int64)
                if self.encode_element:
                    return {
                        #'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'id': tensor_id,
                        'element': element_value
                    }
                else:
                    return {
                        #'latt_dis': tensor_latt_dis,
                        'intensity': tensor_intensity,
                        'id': tensor_id,
                        'element': torch.zeros(92, dtype=torch.int)
                    }              
            cumulative_length += len(db)

    def random_remove_elements(self,lst):
        n = len(lst)
        num_elements_to_remove = random.randint(1, n)
        indices_to_remove = random.sample(range(len(lst)), num_elements_to_remove)
        new_lst = [item for index, item in enumerate(lst) if index not in indices_to_remove]   
        return new_lst

    def mixture(self,xrd,ratio=0.08):
        num = random.randint(1, 100315)
        _row = ase.db.connect(self.db_paths[0][:-8]+'val.db').get(num)
        _int = eval(getattr(_row, 'intensity'))
        result = np.array(xrd) * (1 - ratio) + np.array(_int) * ratio
        return result


    def symbol_to_atomic_number(self,symbol_list):
        # Mapping of element symbols to atomic numbers
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
            'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
            'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
            'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
            'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
            'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
            'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
            'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
            'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
            'Lv': 116, 'Ts': 117, 'Og': 118
        }
        

        atomic_number_list = []

        if symbol_list == []: atomic_number_list.append(0)
        else:
            for symbol in symbol_list:
                if symbol in atomic_numbers:
                    atomic_number_list.append(atomic_numbers[symbol])
                else:
                    atomic_number_list.append(0)  # Append None if symbol not in the dictionary
            
        return atomic_number_list

train_dataset = ASEDataset([os.path.join("/data/zzn/scp", 'trainV.db')],encode_element=False,train=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)

import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.hidden_size = 64
        self.num_layers = 4
        self.gru = nn.GRU(1, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)


        self.fc = nn.Linear(self.hidden_size*2, 100315)

    def forward(self, x):

        x = x.squeeze(1).unsqueeze(-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
import torch.optim as optim
device = torch.device('cuda:5')## Use GPU the train the model
all_labels = []
all_predicted = []
model=BiGRU().to(device)## Move the model to the GPU
optimizer = optim.Adam(model.parameters(), lr=0.00025)## Perform gradient descent using the Adam optimizer
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
for batch_index, data in enumerate(tqdm(train_loader)):
    intensity,labels,element = data['intensity'].to(device), data['id'].to(device), data['element'].to(device)
    # print(labels.max())
    # break
    intensity = intensity.unsqueeze(1)
    element = element.unsqueeze(1)

    batch_size=intensity.shape[0]
    optimizer.zero_grad()


    outputs = model(intensity)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)

    all_labels.extend(labels.cpu().numpy())
    all_predicted.extend(predicted.cpu().numpy())