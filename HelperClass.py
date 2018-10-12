#2016-06-21
#Used for calculating weighted degree centralities

#Import 
import networkx as nx
import pandas as pd
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import pyplot, patches
import numpy as np
import csv
from pylab import imshow, show, cm
import decimal
import re
import scipy

#Quality subjects
#Takes in a QA file and AgeSexFM file and returns a list of quality subjects according to MRI scan number (1 or 2)
def Quality_Control(FM_path, QA_path, MRI_num):
    
    data = pd.DataFrame.from_csv(FM_path)
    subjects = data["Subject"][0:163]
    dQA = pd.DataFrame.from_csv(QA_path)

    subjectsGreat = []; subjectsGreat_tp2 = []; subjectsGreat_tp1 = []
    for subject in subjects:
        if (dQA.loc[subject]["DTI"] < "2" and dQA.loc[subject]["T1"] < "2"):
            subjectsGreat.append(subject)
    for sub in subjectsGreat:
        if (re.search("MRI2",sub)):
            subjectsGreat_tp2.append(sub)
    for sub in subjectsGreat:
        if (re.search("MRI1",sub)):
            subjectsGreat_tp1.append(sub)

    if MRI_num == 1:
        return subjectsGreat_tp1
    if MRI_num == 2:
        return subjectsGreat_tp2

#WDC using fa_mean
def WDC1(G, node):
    edges_fa_mean = nx.get_edge_attributes(G,"fa_mean")
    edges_number_of_fibers = nx.get_edge_attributes(G,"number_of_fibers")
    fa_mean_list = []
    number_of_fibers_list = []
    for i in G.neighbors(node):   
        if node > i:
            fa_mean_list.append(edges_fa_mean[i, node])
            number_of_fibers_list.append(edges_number_of_fibers[i, node])
        if i > node:
            fa_mean_list.append(edges_fa_mean[node,i])
            number_of_fibers_list.append(edges_number_of_fibers[node,i]) 
    multiplication = [a*b for a,b in zip(fa_mean_list,number_of_fibers_list)] 
    summation = sum(multiplication)
    return summation

#WDC using adc_mean
def WDC2(G, node):
    summation = 0
    edges_adc_mean = nx.get_edge_attributes(G,"adc_mean")
    edges_number_of_fibers = nx.get_edge_attributes(G,"number_of_fibers")
    adc_mean_list = []
    number_of_fibers_list = []
    for i in G.neighbors(node):   
        if node > i:
            adc_mean_list.append(edges_adc_mean[i, node])
            number_of_fibers_list.append(edges_number_of_fibers[i, node])
        if i > node:
            adc_mean_list.append(edges_adc_mean[node,i])
            number_of_fibers_list.append(edges_number_of_fibers[node,i]) 
        adc_scaled_list = [ decimal.Decimal(x) ** -2 for x in adc_mean_list ] #raise adc_mean to the negative second power
        multiplication =[a*b for a,b in zip(adc_scaled_list,number_of_fibers_list)] #element wise multiplication
        summation = sum(multiplication) #sum of multiplied elements
    return summation

#WDC using number_of_fibers
def WDC3(G, node):
    summation = 0
    edges_number_of_fibers = nx.get_edge_attributes(G,"number_of_fibers")
    number_of_fibers_list = []
    for i in G.neighbors(node):   
        if node > i:
            number_of_fibers_list.append(edges_number_of_fibers[i, node])
        if i > node:
            number_of_fibers_list.append(edges_number_of_fibers[node,i]) 
        summation = sum(number_of_fibers_list) #sum of multiplied elements
    return summation 

#Creates a square matrix
def build_matrix(a, edge, binarize = False):
    for u,v,d in a.edges_iter(data=True):
        a.edge[u][v]['weight'] = a.edge[u][v][edge]
    bb=nx.to_numpy_matrix(a)
    if binarize:
        c=np.zeros(bb.shape)
        c[bb>0] = 1
        b = c
    else:
        b = bb
    imshow(b, interpolation='nearest', cmap=cm.jet, vmin = b.min(), vmax=b.max())
    plt.colorbar(orientation='vertical')
    show()
    
def efficiency(G):
    avg = 0.0
    n = len(G)
    for node in G:
        path_length=nx.single_source_shortest_path_length(G, node)
        avg += sum(1.0/v for v in path_length.values() if v !=0)
    avg *= 1.0/(n*(n-1))
    return avg