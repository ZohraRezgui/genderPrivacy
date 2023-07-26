
import numpy as np
import os
import time



def get_gender(gender):
    m_idx = (gender == 1.0).nonzero()
    f_idx = (gender == 0.0).nonzero()

    return m_idx, f_idx

def get_unique_gender(labels, gender):
    f_label = [l for (i, l) in enumerate(labels) if gender[i] == 0]
    m_label = [l for (i, l) in enumerate(labels) if gender[i] == 1]

    return np.array(list(set(f_label))), np.array(list(set(m_label)))


def make_weights_for_balanced_classes(samples, target, nclasses):                        
    count = [0] * nclasses                                                      
    for item in target:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(samples)                                              
    for idx, val in enumerate(samples):                                          
        weight[idx] = weight_per_class[target[idx]]                                  
    return weight

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif os.path.exists(dir):
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))
        time.sleep(2)
