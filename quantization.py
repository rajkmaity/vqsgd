import numpy as np
from numpy import linalg as LA

"""
    The idea of vqsgd algorithm is the following 
    Given any data vector 'x'
    x= x/||x||  the norm can  be either l1 or l2 norm
    Now x can be written as 
    x= positive_coeff * I_{d x d} + negative_coeff *(-I_{d x d})
    where I_{d x d} is identity matrix 
    the way [positive_coeff, negative_coeff] are calculated :
    
    1. gamma= 1- ||x||_1/sqrt(d)
    2. for i in range(len(x)):
        if x[i] >0:
            positive_coeff[i]=x[i]/sqrt(d) + gamma/2d
        elif x[i] <0:
            negative_coeff[i]=x[i]/sqrt(d) +gamma/2d
        else:
            positive_coeff[i]=gamma/2d
            negative_coeff[i]=gamma/2d
For more details please consult the paper: https://arxiv.org/abs/1911.07971
"""

def vqsgd_with_l2_norm(data_vector,d,num_index):
    data_vector= np.random.normal(0,1,d)
    data_l2=LA.norm(data_vector)
    normalized_data=data_vector/data_l2
    gamma= 1- (LA.norm(normalized_data,1)/np.sqrt(d))
    positive_coeff=np.zeros(d)
    negative_coeff=np.zeros(d)
    positive_coeff=np.abs(normalized_data)/np.sqrt(d)
    negative_coeff=np.abs(normalized_data)/np.sqrt(d)
    positive_coeff[np.where (normalized_data<0)]=0
    negative_coeff[np.where (normalized_data>0)]=0
    prob=(np.concatenate((positive_coeff,negative_coeff))).flatten()
    prob=prob + (gamma/(2*d))
    pick_index= np.random.choice(2*d, num_index, p=prob)
    return pick_index, data_l2

def vqsgd_with_l1_norm(data_vector,d,num_index):
    data_vector= np.random.normal(0,1,d)
    data_l1=LA.norm(data_vector,1)
    normalized_data=data_vector/data_l1
    gamma= 1- (LA.norm(normalized_data,1)/np.sqrt(d))
    positive_coeff=np.zeros(d)
    negative_coeff=np.zeros(d)
    positive_coeff=np.abs(normalized_data)/np.sqrt(d)
    negative_coeff=np.abs(normalized_data)/np.sqrt(d)
    positive_coeff[np.where (normalized_data<0)]=0
    negative_coeff[np.where (normalized_data>0)]=0
    prob=(np.concatenate((positive_coeff,negative_coeff))).flatten()
    prob=prob + (gamma/(2*d))
    pick_index= np.random.choice(2*d, num_index, p=prob)
    return pick_index, data_l1

### For qsgd please see the paper : https://arxiv.org/abs/1610.02132
def QSGD(data_vector):
    l= len(data_vector)
    data_l2=LA.norm(data_vector,2)
    prob= np.abs(data_vector)/data_l2
    temp= np.zeros(l)
    temp=data_vector
    #temp[np.where(g==0)]=1
    quantized_vector=np.multiply(np.sign(temp),(np.random.uniform(size=l)<prob)*1 )
    return data_l2, quantized_vector
