# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:38:15 2019

@author: worc3920
"""

#dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq

#computation
#import h5py
import numpy as np
import scipy.special.lambertw as lambertw
import math

#plotting
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

#app setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)#,assets_folder="/assets")


server=app.server


#load data
#T=6991
xbar=10
r=34.53,              #0.3*30.39,   #r
beta_c=-2.5087,                #beta_c
beta_d=-0.0766,                 #beta_d
beta_s=np.array([-1.0305, -0.3591, 0.3107, 0.5922, 0.6154, 0.0796, 0.5356, -0.2415,
-0.6286, -1.6736, -0.4351, -0.161, 0, 0.2533, 0.0736, 0.562, 0.2346])
duline=0
dbar=10
imax=101
S=17



#load data
delta1=np.load('assets/WebDelta.npy')
gamma1=np.load('assets/WebGamma.npy')

n=100
gamma=gamma1[:,0::n,:]
delta=delta1[0::n,:]
T=len(range(0,6990,n))

#delta=delta1
#gamma=gamma1
#T=6991

#initial layout
#initX=np.ones((S,1))*5
initX=np.random.randint(0,xbar+1,(S,1)) #initial load state (slider values)
initG=[4,8,13,15] #slots prices initially displayed (switches on)

#styling
sizeToggle=30
heightRow=35

oxblue='#002147'

#auxialiary functions for app layout
def myHeader():
    return [html.Div([
        html.H5(
            children=["Delivery slot"],
            style={'display': 'inline-block','width':'35%'}
            ),
        html.H5(
            children=["Number of orders"],
            style={'display': 'inline-block','width':'45%'}
            ),
        html.H5(
            children=["Plot?"],
            style={'display': 'inline-block','width':'20%','text-align':'right'}
            ),
        ], style={'width': '100%'})]

def mySlot(I,S):
    strSlot=str(I+1).zfill(2)
    strStartTime=str(I+6).zfill(2)
    strEndTime=str(I+7).zfill(2)
    #SLOT 1
    return html.Div([
        html.Div(
            children=["Slot {}: {}:00-{}:00".format(strSlot,strStartTime,strEndTime)],
            style={'display': 'inline-block','width':'35%'}
            ),
        html.Div(
            daq.Slider(
                id='slot{}-slider'.format(strSlot),
                min=0,
                max=xbar,
                marks={xx: '{}'.format('' if I<(S-1) else xx) for xx in range(xbar+1)},
                value=int(initX[I]),
                size='100%',
                color=oxblue
                ),
            style={'display': 'inline-block','width':'45%'}
        ),
        html.Div(
            daq.ToggleSwitch(
                id='slot{}-switch'.format(strSlot),
                value = True if I in initG else False,
                size=sizeToggle,
                color=oxblue,
                style={'float':'right'}
                ),
            style={'display': 'inline-block', 'width':'20%','vertical-align':'top',
                   'float': 'right'}
            ),
        ], style={'width': '100%','height':heightRow})
    

def listSlot(I):
    lis=myHeader()
    for ii in range(I):
        lis.append(mySlot(ii,I))
    return lis

#app layout
app.layout=html.Div([
    #LEFT MARGIN
    html.Div(style={'width': '1%', 'display': 'inline-block'}),
    
    #DATA COLUMN
    html.Div(
        listSlot(S),
        style={'width': '47%', 'display': 'inline-block'}
        ),
        
    #SEPARATOR COLUMN
    html.Div(style={'width': '4%','display': 'inline-block'}),
    
    #PLOT COLUMN
    html.Div([#59% wide column
        #PLOT HEADER
        html.H5(
            children=["Optimised delivery slot prices"],
            style={'width':'100%'}),
        #PLOT
        #html.Div(style={'marginTop':0}),
        dcc.Graph(id='price-fig'),
        #ITERATION HEADER
        html.H5(
            children=["Algorithm iterations"],
            style={'width':'100%'}),
        daq.Slider(
              min=0,
              max=100,
              marks={i: '{}'.format(i) for i in range(0,101,10)},
              value=100,
              size='100%',
              color=oxblue,
              id='iter'
              )
     ], style={'width': '47%','display':'inline-block','vertical-align':'top'})
     ])

#auxiliary callback functions
def myCallbackInput():
    lis=[Input('iter','value')]
    for i in range(S):
        strSlot=str(i+1).zfill(2)
        lis.append(Input('slot{}-slider'.format(strSlot),'value'))
        lis.append(Input('slot{}-switch'.format(strSlot),'value'))
    return lis

#CALLBACKS
@app.callback(
    Output('price-fig','figure'),
    myCallbackInput())
def updateFig(*args):
    it=args[0]+1
    xx=[] #slider, orders x
    g=[] #switch, graphing boolean (plot true/false)
    for i in range(1,2*S,2):
        xx.append(args[i])
    for j in range(2,2*S+1,2):
        g.append(args[j])
    
    xx=np.array(xx).reshape((S,1))
    #compute value at x
    Vx=np.zeros((T,1))
    for tt in range(T):
        Vx[tt]=np.amin(-gamma[:,tt,0:it].reshape(it,S)@xx+delta[tt,0:it].reshape(it,1),axis=0)

    #compute value at neighbours
    Vy=np.zeros((T,S))
    for ss in range(S):
        for tt in range(T):
            x1s=xx+0
            x1s[ss]+=1
            Vy[tt,ss]=np.amin(-gamma[:,tt,0:it].reshape(it,S)@x1s+delta[tt,0:it].reshape(it,1),axis=0)

    #compute opp cost and prices
    oppCost=np.tile(Vx,(1,S))-Vy
    dPlot=np.zeros((T,S))
    F=xx<xbar
    for tt in range(T):
        myGamma=oppCost[tt,:].reshape(S,1)
        myExp = beta_c + beta_s + beta_d * (myGamma - r)
        mySum = sum(np.exp(myExp))
        myin = mySum / np.exp(1)
        h = 1 + np.real(lambertw(myin))
        dStarAdd = np.zeros((S,1))

        for ss in range(S):
            if F[ss] == 0:
                #dStarAdd[ss,0] = math.inf
                dStarAdd[ss,0]=math.inf
    
        dStar = F*(myGamma -r - h.reshape(S,1) / beta_d)
        dStar = np.maximum(np.ones((S,1))*duline, dStar)
        dStar = np.minimum(np.ones((S,1))*dbar, dStar)
        for ss in range(S):
            if np.isnan(dStar[ss]):
                dStar[ss]=math.inf
        dStar=dStar+dStarAdd
        dPlot[tt,:]=dStar.reshape(S,)
    dPlot[dPlot==math.inf]=10
    plotSlots=[]
    for i in range(S):
        if g[i]==True:
            plotSlots.append(i)
    traces=[]
    for ss in plotSlots:
        traces.append(dict(
            x=22-22*(1+np.array(range(0,6990,n)))/range(0,6990,n)[-1],
            y=dPlot[:,ss],
            text="Slot {}".format(str(ss+1).zfill(2)),
            name="Slot {}".format(str(ss+1).zfill(2))
            ))
        
    figure=dict(
        data= traces,
        layout= dict(
            margin={'l': 30, 'b': 30, 't': 30, 'r': 30},
            hovermode='closest',
            height=520,
            transition = {'duration': 500},
            xaxis={'title':'Days before delivery day',
                   'range':[22,0]},
            yaxis={'range':[0,10],'title':'Delivery price in £'}
            )
        )
    return figure
            
if __name__ == '__main__':
    app.run_server(debug=True)