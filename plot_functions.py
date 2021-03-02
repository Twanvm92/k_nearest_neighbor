# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:23:08 2021

@author: 20172458
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def bar_plot(question, title, subtitle, variable_df, bar_width_scaler, text_label):
    
    plt.figure(figsize=(15, 7))
    #label_font = {'fontname': 'calibri'}
    other_font = {'fontname': 'arial'}
    
    k = variable_df['k']
    x_axis = np.arange(len(variable_df)) 
    variable_df_excl = variable_df.loc[:, variable_df.columns != 'k']
    variable_names = list(variable_df_excl.columns)
    nr_var = len(variable_names)
    width = 1 / (bar_width_scaler * nr_var)
    tot_width = nr_var * width 
    
    for name in variable_names:
        rects = plt.bar(x_axis + variable_names.index(name) * (width), 
                      variable_df_excl[name], width, label = name)
        
        if text_label:
            for rect in rects:
                height = rect.get_height()
                plt.annotate('{}'.format(height),
                            fontsize = 11,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
    
    # Formatting
    plt.xticks(x_axis + ((tot_width - width) / 2), k)
    plt.ylabel('Risk', fontsize = 16, ** other_font)
    plt.xlabel('k', fontsize = 16, ** other_font)
    plt.title(title, loc='left', fontsize=20, fontweight = "bold", ** other_font)
    plt.title(subtitle, loc='right', fontsize=14, color='dimgray', ** other_font)
    plt.legend(loc = 1, edgecolor = 'white')
    plt.savefig(question + ".jpg")

def line_plot(question, title, subtitle, variable_df, line_thick, filled):
    
    plt.figure(figsize=(15, 7))
    #label_font = {'fontname': 'calibri'}
    other_font = {'fontname': 'arial'}
    
    k = variable_df['k']
    variable_df_excl = variable_df.loc[:, variable_df.columns != 'k']
    variable_names = list(variable_df_excl.columns)
    
    for name in variable_names:
        if filled:
            plt.fill_between(k, variable_df_excl[name], label = name, alpha = 0.3)
        else:
            plt.plot(k, variable_df_excl[name], label = name, linewidth = line_thick)
            plt.scatter(k, variable_df_excl[name], s = 5 * line_thick)
        
    # Formatting
    plt.xticks(k)
    plt.ylabel('Risk', fontsize = 16, ** other_font)
    plt.xlabel('k', fontsize = 16, ** other_font)
    plt.title(title, loc='left', fontsize=20, fontweight = "bold", ** other_font)
    plt.title(subtitle, loc='right', fontsize=14, color='dimgray', ** other_font)
    plt.legend(loc = 1, edgecolor = 'white')
    plt.savefig(question + ".jpg")
    
    
# EXAMPLES
df = pd.read_csv("results_q1c_plot.csv")

#bar_plot('Qa', 'Empirical training and test risk', 'Euclidean distance', df, 1.5, False)
line_plot('Qc', 'Empirical test risk', 'Minkowski distance', df, 1.2,False)