#Import 
import networkx as nx
import pandas as pd
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import pyplot, patches
import numpy as np
import csv
from pylab import imshow, show, cm, plot
import decimal
import re
import scipy
from tvtk.api import tvtk
from HelperClass import *
from pandas import DataFrame
#%matplotlib inline
import statsmodels.api as sm
#import pylab as pylab
import pandas.stats.api as pdstat
import patsy
import statsmodels.formula.api as smFormula
import scipy.stats as stats
import math
#import community
import matplotlib as ml
import pylab
#from NetMets2 import *
from scipy.optimize import curve_fit
from numpy import arange,array,ones#,random,linalg
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model
#ml.style.use('ggplot')

#Master Class for all methods
class Masterclass(object):  
    
    def __init__(self, gpickle_path, FM_path, QA_path, subject_path):
        self.gpickle_path = gpickle_path
        self.FM_path = FM_path
        self.QA_path = QA_path
        self.subject_path = subject_path

    #Description: Graphs all edges and nodes with labels
    #Input: gpickle
    #Output: Matlab image   
    def grapher(self):
        g = nx.read_gpickle(self.subject_path)
        g.remove_node(83)
        position_key = "dn_position"
        edge_key = "fa_mean"
        node_label_key = "dn_fsname"

        create_label = g.nodes()
        #nodes of interest: 8, 18, 35, 37, 49, 59, 76, 78
        nr_nodes = len(create_label)
        position_array = np.zeros( (nr_nodes, 3) )
        val_array = np.zeros((nr_nodes, 1))
        for i,nodeid in enumerate(create_label):
            #gives position of certain node
            pos = g.node[nodeid][position_key]
            pos = np.array(pos)
            position_array[i,:] = pos
        x, y, z = position_array[:,0], position_array[:,1], position_array[:,2]
        edges = np.array(g.edges())
        nr_edges = len(edges)
        ev = np.zeros( (nr_edges, 1) )
        for i,d in enumerate(g.edges_iter(data=True)):
            ev[i] = d[2][edge_key]
            assert d[0] == edges[i,0] and d[1] == edges[i,1]
        edges=edges - 1
        start_positions = position_array[edges[:, 0], :].T
        end_positions = position_array[edges[:, 1], :].T
        vectors = end_positions - start_positions

            # Creates Edges
        nodesource = mlab.pipeline.scalar_scatter(x, y, z, name = 'Node Source')
        nodes = mlab.pipeline.glyph(nodesource, scale_factor=3.0, scale_mode='none', name = 'Nodes', mode='cube')
        nodes.glyph.color_mode = 'color_by_scalar'
        vectorsrc = mlab.pipeline.vector_scatter(start_positions[0], 
                                         start_positions[1],
                                         start_positions[2],
                                         vectors[0],
                                         vectors[1],
                                         vectors[2],
                                         name = 'Connectivity Source')
        da = tvtk.DoubleArray(name=edge_key)
        da.from_array(ev)
        vectorsrc.mlab_source.dataset.point_data.add_array(da)
        vectorsrc.mlab_source.dataset.point_data.scalars = da.to_array()
        vectorsrc.mlab_source.dataset.point_data.scalars.name = edge_key
        vectorsrc.outputs[0].update()
        thres = mlab.pipeline.threshold(vectorsrc, name="Thresholding")

        #control appearance of edges and nodes (color vs size)
        #how to mix and match color and size differences
        myvectors = mlab.pipeline.vectors(thres,colormap='OrRd',
                                                        #mode='cylinder',
                                                        name='Connections',
                                                        scale_factor=1,
                                                        resolution=20,
                                                        # make the opacity of the actor depend on the scalar.
                                                        transparent=True,
                                                        scale_mode = 'vector')

        myvectors.glyph.glyph_source.glyph_source.glyph_type = 'dash'
        myvectors.glyph.color_mode = 'color_by_scalar'
        myvectors.glyph.glyph.clamping = False

        # create labels
        for la in create_label:
            row_index = la - 1
            label = g.node[la][node_label_key]
            mlab.text3d(position_array[row_index,0],
                            position_array[row_index,1],
                            position_array[row_index,2],
                            '     ' + label,
                            name = 'Node ' + label,
                            color = (0,0,0),
                            scale = 3,
                            line_width = 3)

        #Scales the nodes
        for i,nodeid in enumerate(g.nodes()):
                #nodes of interest: 8, 18, 35, 37, 49, 59, 76, 78
                #if nodeid==8 or nodeid==18 or nodeid==35 or nodeid==37 or nodeid==49 or nodeid==59 or nodeid==76 or nodeid==78:
                pos = g.node[nodeid][position_key]
                val = g.degree(nodeid)
                pos = np.array(pos)
                val = np.array(val)
                position_array[i,:] = pos
                val_array[i] = val

        x, y, z, value = position_array[:,0], position_array[:,1], position_array[:,2], val_array[:,0]
        #mlab.figure()
        mlab.points3d(x, y, z, value)
        mlab.show()
        
    

    #Converts csv files into text files that are compatible for BrainNet
    #Input: gpickle, Quality information, csv files)
    #Output: text file
        #Rich Club nodes are highlighted
    def BrainNet_Node(self, node_file_path, data_file, subject_num):
        #Desired graph
        G = nx.read_gpickle(self.subject_path)

        #Manipulation of graph
        G.remove_node(83)
        position_key = "dn_position"
        edge_key = "fa_mean"
        node_label_key = "dn_name"
        create_label = G.nodes()
        nr_nodes = len(G.nodes())
        position_array = np.zeros((nr_nodes, 3))
        
        #New text file
        f = open(node_file_path, 'w')
        
        dQA = pd.DataFrame.from_csv(self.QA_path)
        data = pd.DataFrame.from_csv(self.FM_path)
        subjects = data["Subject"][0:163]
        subjectsGreat = []
        subjectsGreat_tp2 = []
        subjectsGreat_tp1 = []
        for subject in subjects:
            if (dQA.loc[subject]["DTI"] < "2" and dQA.loc[subject]["T1"] < "2"):
                subjectsGreat.append(subject)
        for sub in subjectsGreat:
            if (re.search("MRI2",sub)):
                subjectsGreat_tp2.append(sub)
        for sub in subjectsGreat:
            if (re.search("MRI1",sub)):
                subjectsGreat_tp1.append(sub)
        
        dfWDC1 = pd.DataFrame.from_csv(data_file)
        
        target = subjects.iloc[subject_num]
        
        color_num = 0
        
        for i, nodeid in enumerate(create_label):
            pos = G.node[nodeid][position_key]
            pos = np.array(pos)
            position_array[i,:] = pos    
            name = G.node[nodeid]["dn_name"]
            if nodeid == 8 or nodeid == 18 or nodeid == 35 or nodeid == 37 or nodeid == 49 or nodeid == 59 or nodeid == 76 or nodeid ==78:
                color_num = 1
            else:
                color_num = 3
            f.write(str(-1 * position_array[i,2])+"\t"+str(-1 * position_array[i,0])+"\t"+str(position_array[i,1])+"\t"+str(color_num)+"\t"+str(dfWDC1.loc[target]["WDC1" + name])+"\t"+G.node[nodeid]["dn_name"]+"\n")
            
    def BrainNet_Edge(self, edge_text_file, data_file):
        edge_data = pd.read_csv(data_file, sep = ",", header = None)        
        np.savetxt(edge_text_file, edge_data, delimiter= "\t")
        
    #Creates csv files that contain data of weighted centrality for edges divided by subject group
    #Input: gpickle, Quality information, FM path
    #Output: 10 CSV files related to WC for five different subject groups (one batch that is int, one batch that is raw data)    
    def WC_Edge(self, output_path):     
        data = pd.DataFrame.from_csv(self.FM_path)
        age = data["Age"]
        gender = data["Gender"]
        subjects = data["Subject"][0:163]
        gpickle_path=(self.gpickle_path)
        
        dQA = pd.DataFrame.from_csv(self.QA_path)
        data = pd.DataFrame.from_csv(self.FM_path)
        subjects = data["Subject"][0:163]
        dQA = pd.DataFrame.from_csv(self.QA_path)
        subjectsGreat = []
        subjectsGreat_tp2 = []
        subjectsGreat_tp1 = []
        for subject in subjects:
            if (dQA.loc[subject]["DTI"] < "2" and dQA.loc[subject]["T1"] < "2"):
                subjectsGreat.append(subject)
        for sub in subjectsGreat:
            if (re.search("MRI2",sub)):
                subjectsGreat_tp2.append(sub)
        for sub in subjectsGreat:
            if (re.search("MRI1",sub)):
                subjectsGreat_tp1.append(sub)
                
        G_fa_av = np.zeros([82,82]); B_fa_av = np.zeros([82,82]); S_fa_av = np.zeros([82,82]); M_fa_av = np.zeros([82,82]); L_fa_av = np.zeros([82,82]);
        G_fib = np.zeros([82,82]); B_fib = np.zeros([82,82]); S_fib = np.zeros([82,82]); M_fib = np.zeros([82,82]); L_fib =  np.zeros([82,82]);
        counterG = 0; counterB = 0; counter8 = 0; counter10 = 0; counter13 = 0;

        for sub in subjectsGreat_tp1:
            G = nx.read_gpickle(gpickle_path+"/connectome_scale33_"+sub+".gpickle")
            G.remove_node(83)
            d = {}
            d[sub] = {}
            d[sub]["age"] = data[data.Subject == sub]["Age"].item()
            d[sub]["sex"] = data[data.Subject == sub]["Gender"].item()

            #fa mean
            if d[sub]["sex"] == "F":
                G_fa = nx.to_numpy_matrix(G , weight = "fa_mean")
                counterG = counterG  + 1
                G_fa_av = G_fa_av + G_fa
            if d[sub]["sex"] == "M":
                B_fa = nx.to_numpy_matrix(G , weight = "fa_mean")
                counterB = counterB  + 1
                B_fa_av = B_fa_av + B_fa
            if d[sub]["age"] < 10:
                S_fa = nx.to_numpy_matrix(G , weight = "fa_mean")
                counter8 = counter8  + 1
                S_fa_av = S_fa_av + S_fa
            if  10 <= d[sub]["age"] < 13:
                M_fa = nx.to_numpy_matrix(G , weight = "fa_mean")
                counter10 = counter10  + 1
                M_fa_av = M_fa_av + M_fa
            if d[sub]["age"] >= 13:
                L_fa = nx.to_numpy_matrix(G , weight = "fa_mean")
                counter13 = counter13  + 1
                L_fa_av = L_fa_av + L_fa

            #Number of Fibers
            if d[sub]["sex"] == "F":
                G_num = nx.to_numpy_matrix(G , weight = "number_of_fibers")
                G_fib = G_fib + G_num
            if d[sub]["sex"] == "M":
                B_num = nx.to_numpy_matrix(G , weight = "number_of_fibers")
                B_fib = B_fib + B_num
            if d[sub]["age"] < 10:
                S_num = nx.to_numpy_matrix(G , weight = "number_of_fibers")
                S_fib = S_fib + S_num
            if  10 < d[sub]["age"] < 13:
                M_num = nx.to_numpy_matrix(G , weight = "number_of_fibers")
                M_fib = M_fib + M_num
            if d[sub]["age"] >= 13:
                L_num = nx.to_numpy_matrix(G , weight = "number_of_fibers")
                L_fib = L_fib + L_num        

        G_fa_av = G_fa_av/counterG
        B_fa_av = B_fa_av/counterB
        S_fa_av = S_fa_av/counter8
        M_fa_av = M_fa_av/counter10
        L_fa_av = L_fa_av/counter13

        G_fib = G_fib/counterG
        B_fib = B_fib/counterB
        S_fib = S_fib/counter8
        M_fib = M_fib/counter10
        L_fib = L_fib/counter13

        G_wc1 = np.multiply(G_fa_av,G_fib)
        B_wc1 = np.multiply(B_fa_av,B_fib) 
        S_wc1 = np.multiply(S_fa_av,S_fib) 
        M_wc1 = np.multiply(M_fa_av,M_fib) 
        L_wc1 = np.multiply(L_fa_av,L_fib) 

        G_wc1_int = G_wc1.astype(int)
        B_wc1_int = B_wc1.astype(int)
        S_wc1_int = S_wc1.astype(int)
        M_wc1_int = M_wc1.astype(int)
        L_wc1_int = L_wc1.astype(int)

        pd.DataFrame.to_csv(pd.DataFrame(G_wc1), output_path + "/G_wdc1.csv")
        pd.DataFrame.to_csv(pd.DataFrame(B_wc1), output_path + "/B_wdc1.csv")
        pd.DataFrame.to_csv(pd.DataFrame(S_wc1), output_path + "/S_wdc1.csv")
        pd.DataFrame.to_csv(pd.DataFrame(M_wc1), output_path + "/M_wdc1.csv")
        pd.DataFrame.to_csv(pd.DataFrame(L_wc1), output_path + "/L_wdc1.csv")

        pd.DataFrame.to_csv(pd.DataFrame(G_wc1_int), output_path + "/G_wdc1_int.csv")
        pd.DataFrame.to_csv(pd.DataFrame(B_wc1_int), output_path + "/B_wdc1_int.csv")
        pd.DataFrame.to_csv(pd.DataFrame(S_wc1_int), output_path + "/S_wdc1_int.csv")
        pd.DataFrame.to_csv(pd.DataFrame(M_wc1_int), output_path + "/M_wdc1_int.csv")
        pd.DataFrame.to_csv(pd.DataFrame(L_wc1_int), output_path + "/L_wdc1_int.csv")
        
    #Creates csv files that contain data of weighted degree centrality for nodes
    #Input: gpickle, Quality path, FM path
    #Output: 5 WDC csv files related by subject group
    def WC_Node(self, output_path):        
        dictNetMetWDCNoBS_one = {}; dictNetMetWDCNoBS_two = {}; dictNetMetWDCNoBS_three = {}
        counterG = 0; counterB = 0; counter8 = 0; counter10 = 0; counter13 = 0; counter = 0;
        
        data = pd.DataFrame.from_csv(self.FM_path)
        age = data["Age"]
        gender = data["Gender"]
        subjects = data["Subject"][0:163]
        
        for sub in subjects:
            avg = 0;
            G = nx.read_gpickle(self.gpickle_path+"/connectome_scale33_"+sub+".gpickle")
            G.remove_node(83)
            d_one = {}; d_two = {}; d_three = {}
            d_one[sub] = {}; d_two[sub] = {}; d_three[sub] = {}
            d_one[sub]["age"] = data[data.Subject == sub]["Age"].item(); d_two[sub]["age"] = data[data.Subject == sub]["Age"].item(); d_three[sub]["age"] = data[data.Subject == sub]["Age"].item();  
            d_one[sub]["sex"] = data[data.Subject == sub]["Gender"].item(); d_two[sub]["age"] = data[data.Subject == sub]["Age"].item(); d_three[sub]["age"] = data[data.Subject == sub]["Age"].item(); 

            ROInumR = []; ROInumL = []; ROInamR = []; ROInamL = []
            for node in nx.nodes(G):
                if node < 42:
                    ROInumR.append(node)
                    ROInamR.append(G.node[node]["dn_name"])
                if 41 < node < 83:
                    ROInumL.append(node)
                    ROInamL.append(G.node[node]["dn_name"])
            zipped = zip( ROInamR, ROInumR, ROInamL, ROInumL)
            for zippy in zipped:

                yvarR = WDC1(G,zippy[1])
                yvarL = WDC1(G,zippy[3])
                avg = avg + yvarR + yvarL
                d_one[sub]["WDC1"+zippy[0]] = yvarR
                d_one[sub]["WDC1"+zippy[2]] = yvarL

                yvarR = WDC2(G,zippy[1])
                yvarL = WDC2(G,zippy[3])
                d_two[sub]["WDC2"+zippy[0]] = yvarR
                d_two[sub]["WDC2"+zippy[2]] = yvarL

                yvarR = WDC3(G,zippy[1])
                yvarL = WDC3(G,zippy[3])
                d_three[sub]["WDC3"+zippy[0]] = yvarR
                d_three[sub]["WDC3"+zippy[2]] = yvarL
                
            counter += 1
            avg = avg/counter
            d_one[sub]["WDC1Global"] = avg

            dictNetMetWDCNoBS_one.update(d_one)
            dictNetMetWDCNoBS_two.update(d_two)
            dictNetMetWDCNoBS_three.update(d_three)
        
        df_one = pd.DataFrame.from_dict(dictNetMetWDCNoBS_one)
        df_two = pd.DataFrame.from_dict(dictNetMetWDCNoBS_two)
        df_three = pd.DataFrame.from_dict(dictNetMetWDCNoBS_three)  

        df_one = df_one.transpose()
        df_two = df_two.transpose()
        df_three = df_three.transpose()
        
        pd.DataFrame.to_csv(pd.DataFrame(df_one), output_path + "/wc1_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_two), output_path + "/wc2_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_three), output_path + "/wc3_node.csv")
        
        df_one = df_one.transpose()
        df_two = df_two.transpose()
        df_three = df_three.transpose()

        for sub in subjects:
            if df_one[sub]["sex"] == "F":
                counterG = counterG + 1
            if df_one[sub]["sex"] == "M":
                counterB = counterB + 1
            if df_one[sub]["age"] <= 10:
                counter8 = counter8 + 1
            if 10 < df_one[sub]["age"] < 13:
                counter10 = counter10 + 1
            if df_one[sub]["age"] >= 13:
                counter13 = counter13 + 1

        df_girl_1 = pd.DataFrame(data=None)
        df_boy_1 = pd.DataFrame(data=None)
        df_small_1 = pd.DataFrame(data=None)
        df_medium_1 = pd.DataFrame(data=None)
        df_large_1 = pd.DataFrame(data=None)

        df_girl_2 = pd.DataFrame(data=None)
        df_boy_2 = pd.DataFrame(data=None)
        df_small_2 = pd.DataFrame(data=None)
        df_medium_2 = pd.DataFrame(data=None)
        df_large_2 = pd.DataFrame(data=None)

        df_girl_3 = pd.DataFrame(data=None)
        df_boy_3 = pd.DataFrame(data=None)
        df_small_3 = pd.DataFrame(data=None)
        df_medium_3 = pd.DataFrame(data=None)
        df_large_3 = pd.DataFrame(data=None)

        d_girl_1 = {}
        d_boy_1 = {}
        d_small_1 = {}
        d_medium_1 = {}
        d_large_1 = {}

        d_girl_2 = {}
        d_boy_2 = {}
        d_small_2 = {}
        d_medium_2 = {}
        d_large_2 = {}

        d_girl_3 = {}
        d_boy_3 = {}
        d_small_3 = {}
        d_medium_3 = {}
        d_large_3 = {}

        for sub in subjects:
            if df_one[sub]["sex"] == "F":
                d_girl_1.update({sub:df_one[sub]})
                d_girl_2.update({sub:df_two[sub]})
                d_girl_3.update({sub:df_three[sub]})
            if df_one[sub]["sex"] == "M":
                d_boy_1.update({sub:df_one[sub]})
                d_boy_2.update({sub:df_two[sub]})
                d_boy_3.update({sub:df_three[sub]})
            if df_one[sub]["age"] >= 10:
                d_small_1.update({sub:df_one[sub]})
                d_small_2.update({sub:df_two[sub]})
                d_small_3.update({sub:df_three[sub]})
            if 10 < df_one[sub]["age"] < 13:
                d_medium_1.update({sub:df_one[sub]})
                d_medium_2.update({sub:df_two[sub]})
                d_medium_3.update({sub:df_three[sub]})
            if df_one[sub]["age"] >= 13:
                d_large_1.update({sub:df_one[sub]})
                d_large_2.update({sub:df_two[sub]})
                d_large_3.update({sub:df_three[sub]})

        for value in d_girl_1.values():
            if type(value) == int:
                value = value/counterG
        for value in d_girl_2.values():
            if type(value) == int:
                value = value/counterG
        for value in d_girl_3.values():
            if type(value) == int:
                value = value/counterG

        for value in d_boy_1.values():
            if type(value) == int:
                value = value/counterB
        for value in d_boy_2.values():
            if type(value) == int:
                value = value/counterB
        for value in d_boy_3.values():
            if type(value) == int:
                value = value/counterB

        for value in d_small_1.values():
            if type(value) == int:
                value = value/counter8
        for value in d_small_2.values():
            if type(value) == int:
                value = value/counter8
        for value in d_small_3.values():
            if type(value) == int:
                value = value/counter8

        for value in d_medium_1.values():
            if type(value) == int:
                value = value/counter10
        for value in d_medium_2.values():
            if type(value) == int:
                value = value/counter10
        for value in d_medium_3.values():
            if type(value) == int:
                value = value/counter10

        for value in d_large_1.values():
            if type(value) == int:
                value = value/counter13
        for value in d_large_2.values():
            if type(value) == int:
                value = value/counter13
        for value in d_large_3.values():
            if type(value) == int:
                value = value/counter13

        df_girl_1 = pd.DataFrame.from_dict(d_girl_1)
        df_boy_1 = pd.DataFrame.from_dict(d_boy_1)
        df_small_1 = pd.DataFrame.from_dict(d_small_1)
        df_medium_1 = pd.DataFrame.from_dict(d_medium_1)
        df_large_1 = pd.DataFrame.from_dict(d_large_1)

        df_girl_2 = pd.DataFrame.from_dict(d_girl_2)
        df_boy_2 = pd.DataFrame.from_dict(d_boy_2)
        df_small_2 = pd.DataFrame.from_dict(d_small_2)
        df_medium_2 = pd.DataFrame.from_dict(d_medium_2)
        df_large_2 = pd.DataFrame.from_dict(d_large_2)

        df_girl_3 = pd.DataFrame.from_dict(d_girl_3)
        df_boy_3 = pd.DataFrame.from_dict(d_boy_3)
        df_small_3 = pd.DataFrame.from_dict(d_small_3)
        df_medium_3 = pd.DataFrame.from_dict(d_medium_3)
        df_large_3 = pd.DataFrame.from_dict(d_large_3)
        
        df_girl_1 = df_girl_1.transpose()
        df_boy_1 = df_boy_1.transpose()
        df_small_1 = df_small_1.transpose()
        df_medium_1 = df_medium_1.transpose()
        df_large_1 = df_large_1.transpose()
        
        df_girl_2 = df_girl_2.transpose()
        df_boy_2 = df_boy_2.transpose()
        df_small_2 = df_small_2.transpose()
        df_medium_2 = df_medium_2.transpose()
        df_large_2 = df_large_2.transpose()
        
        df_girl_3 = df_girl_3.transpose()
        df_boy_3 = df_boy_3.transpose()
        df_small_3 = df_small_3.transpose()
        df_medium_3 = df_medium_3.transpose()
        df_large_3 = df_large_3.transpose()

        pd.DataFrame.to_csv(pd.DataFrame(df_girl_1), output_path + "/G_wc1_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_boy_1), output_path + "/B_wc1_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_small_1), output_path + "/S_wc1_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_medium_1), output_path + "/M_wc1_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_large_1), output_path + "/L_wc1_node.csv")

        pd.DataFrame.to_csv(pd.DataFrame(df_girl_2), output_path + "/G_wc2_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_boy_2), output_path + "/B_wc2_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_small_2), output_path + "/S_wc2_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_medium_2), output_path + "/M_wc2_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_large_2), output_path + "/L_wc2_node.csv")

        pd.DataFrame.to_csv(pd.DataFrame(df_girl_3), output_path + "/G_wc3_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_boy_3), output_path + "/B_wc3_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_small_3), output_path + "/S_wc3_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_medium_3), output_path + "/M_wc3_node.csv")
        pd.DataFrame.to_csv(pd.DataFrame(df_large_3), output_path + "/L_wc3_node.csv")
        
    def Stats(self, csv_path, output_path):
        dfNoBST = DataFrame.from_csv(csv_path)
        yGLMs = list(dfNoBST.columns.values) # A list of all the y predictors
        list.remove(yGLMs, 'sex'); list.remove(yGLMs, 'age'); 
        #list.remove(yGLMs, 'sexF1M2')
        dfstats = DataFrame()
        dfstatsi = DataFrame()
        for i, yGLM in enumerate(yGLMs):
        #for i in np.arange(0,2): 
            dat = dfNoBST[['age', 'sex']]
            dat['y'] = dfNoBST[yGLM]
            outmodel = smFormula.ols('y ~ age + sex + age*sex', dat).fit()
            CI = outmodel.conf_int()
            dfstatsi['yGLM'] = [yGLM]; dfstatsi['f_stat_p'] = outmodel.f_pvalue; dfstatsi['p-inter'] = outmodel.pvalues[0]; dfstatsi['p-sex[T.M]'] = outmodel.pvalues[1]; dfstatsi['p-age'] = outmodel.pvalues[2]; dfstatsi['p-agebysex'] = outmodel.pvalues[3]; dfstatsi['b-inter'] = outmodel.params[0]; dfstatsi['b-sex[T.M]'] = outmodel.params[1]; dfstatsi['b-age'] = outmodel.params[2]; dfstatsi['b-agebysex'] = outmodel.params[3]; dfstatsi['bse-inter'] = outmodel.bse[0]; dfstatsi['bse-sex[T.M]'] = outmodel.bse[1]; dfstatsi['bse-age'] = outmodel.bse[2]; dfstatsi['bse-agebysex'] = outmodel.bse[3]; dfstatsi['t_intercept'] = outmodel.tvalues[0]; dfstatsi['t_sex[T.M]'] = outmodel.tvalues[1]; dfstatsi['t_age'] = outmodel.tvalues[2]; dfstatsi['t_agebysex'] = outmodel.tvalues[3]; dfstatsi['r^2'] = outmodel.rsquared; dfstatsi['|r|'] = math.sqrt(outmodel.rsquared); dfstatsi['CI_Intercept_low'] = CI.iat[0,0]; dfstatsi['CI_Intercept_high'] = CI.iat[0,1]; dfstatsi['CI_sex_low'] = CI.iat[1,0]; dfstatsi['CI_sex_high'] = CI.iat[1,1]; dfstatsi['CI_age_low'] = CI.iat[2,0]; dfstatsi['CI_age_high'] = CI.iat[2,1]; dfstatsi['CI_agebysex_low'] = CI.iat[3,0]; dfstatsi['CI_agebysex_high'] = CI.iat[3,1]
            dfstats = dfstats.append(dfstatsi,ignore_index=True)
        #    res = outmodel.resid
        #    fig = sm.qqplot(res, stats.t, distargs=(4,))
            plt.show()
        DataFrame.to_csv(dfstats, output_path)
        
    def Stats_Plot(self, csv_path, brain_region):
        brain_region_str = str(brain_region)
        dfNoBS = DataFrame.from_csv(csv_path)
        ageBoyss  = dfNoBS.query('sex == "M"')['age']
        ageGirlss = dfNoBS.query('sex == "F"')['age']
        x = dfNoBS['age']

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle(brain_region_str, fontsize=22 )
        ax.set_xlabel('Age(yrs)', fontsize=14)
        ax.text(0.02,.98,'Girls',verticalalignment='top', horizontalalignment='left',transform=ax.transAxes,color='red', fontsize=12)
        ax.text(0.02,.9,'Boys',verticalalignment='top', horizontalalignment='left',transform=ax.transAxes,color='blue', fontsize=12)
        y = dfNoBS['WDC1rh.superiorfrontal']
        yGirlss = dfNoBS.query('sex == "F"')[brain_region_str]
        yBoyss  = dfNoBS.query('sex == "M"')[brain_region_str]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        slopeG, interceptG, r_valueG, p_valueG, std_errG = stats.linregress(ageGirlss,yGirlss)
        slopeB, interceptB, r_valueB, p_valueB, std_errB = stats.linregress(ageBoyss, yBoyss)
        line = slope*x+intercept;lineG = slopeG*ageGirlss+interceptG;lineB = slopeB*ageBoyss+interceptB
        plt.text(0.99, 0.1, 'r = '+str(round(dfstats.query('yGLM == brain_region_str')["|r|"].item(),2))+"\n p = "+str(round(dfstats.query('yGLM == ""')["f_stat_p"].item(),8))+", N= 121", ha='right', va='center', fontsize="15", transform=ax.transAxes)
        plot(x,line,'k-',ageGirlss,lineG,'r-',ageBoyss,lineB,'b-',ageGirlss,yGirlss,'ro',ageBoyss, yBoyss, 'bo')
        #plt.ylim(0.5,0.78)
        plt.savefig("/Users/KevinLee/Documents/UCI_Research/Stats/" + brain_region_str + ".eps", format='eps')
        plt.show()