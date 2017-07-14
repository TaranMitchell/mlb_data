#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:19:39 2017

@author: taran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
 
# Assign the filename: file
file = 'bumgarnerData2016.csv'
 
# Read the file into a DataFrame: df
df = pd.read_csv(file)

state = df.loc[:,['startingBalls','startingStrikes','endBalls','endStrikes','hitLocation']]

state['startState']=state['startingBalls'].astype(str)+'-'+state['startingStrikes'].astype(str)

#state['IP']=np.where(state['hitLocation']==0,'False','True')
state['IP']=['False' if x == 0 else 'True' for x in state['hitLocation']]

#state['BB']=np.where(state['endBalls']==4,'True','False')
state['BB']=['True' if x == 4 else 'False' for x in state['endBalls']]

#state['K']=np.where(state['endStrikes']==3,'True','False')
state['K']=['True' if x == 3 else 'False' for x in state['endStrikes']]

# create function to determine the end state: endState
def endState(row):
    ''' function that deteremines the outcome of the pitch and creates new state = endState'''
    if row['BB'] == 'True':
        val = 'BB'
    elif row['K'] == 'True':
        val = 'K'
    elif row['IP'] == 'True':
        val = 'IP'
    else:
        val = str(row['endBalls'])+'-'+str(row['endStrikes'])
    return val

# Add endState to the state frame
state['endState'] = state.apply(endState, axis=1)

transition_df = state.loc[:,['startState','endState']]

state_pivot = transition_df.pivot_table(index=['startState'], columns='endState', aggfunc=len, fill_value=0, margins=True, margins_name='Total')

table2 = state_pivot.div(state_pivot.iloc[:,-1], axis=0 )

clean = table2.iloc[:-1,:-1]

extra_rows = pd.DataFrame({'0-1': [0,0,0],
'0-2': [0,0,0],
'1-0': [0,0,0],
'1-1': [0,0,0],
'1-2': [0,0,0],
'2-0': [0,0,0],
'2-1': [0,0,0],
'2-2': [0,0,0],
'3-0': [0,0,0],
'3-1': [0,0,0],
'3-2': [0,0,0],
'BB': [1,0,0],
'IP': [0,1,0],
'K': [0,0,1]},
index=['BB', 'IP', 'K'])

result = clean.append(extra_rows)

# missing 0-0 as end state so that the matrix is 15x15 square
result['0-0'] = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], index=result.index)

#manually reorder columns
result = result[['0-0','0-1','0-2','1-0','1-1','1-2','2-0','2-1','2-2','3-0','3-1','3-2','BB','IP','K']]

# I is an r-by-r identity matrix, 0 is an r-by-t zero matrix, R is a
# nonzero t-by-r matrix, and Q is an t-by-t matrix

#Transient matrix Q
Q = result.iloc[0:12,0:12]
matrixQ = np.matrix(Q)

# Absorption matrix R
R = result.iloc[0:12,12:15]
matrixR = np.matrix(R)

# Transient Identity matrix I
I = np.identity(12)
matrixI = np.matrix(I)

#fundamental matrix N = (I − Q)^-1
test = matrixI - matrixQ
N = inv(np.matrix(test))
matrixN = np.matrix(N)

# Absorption matrix B = NR
B = matrixN * matrixR

c = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
matrixc = np.matrix(c)

t = matrixN * matrixc
matrixt = np.matrix(t)

# Final tables
absorption = pd.DataFrame(data=B, columns=['BB', 'IP', 'K'], index=['0-0','0-1','0-2','1-0','1-1','1-2','2-0','2-1','2-2','3-0','3-1','3-2'])
pitchCount = pd.DataFrame(data=matrixt, columns=['pitch count to absorption'],index=['0-0','0-1','0-2','1-0','1-1','1-2','2-0','2-1','2-2','3-0','3-1','3-2'])

print absorption

absorption.plot.bar(stacked=True)
plt.show()
