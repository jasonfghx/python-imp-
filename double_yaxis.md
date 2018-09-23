import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('Y values for exp(-x)')
ax1.set_title("Double Y axis")

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, 'r')
ax2.set_xlim([0, np.e])
ax2.set_ylabel('Y values for ln(x)')
ax2.set_xlabel('Same X for both exp(-x) and ln(x)')

plt.show()

========================================================

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data1.index, data1["original cond (mS/cm)"], color='green', label='原水電導度')
ax1.plot(data1.index, data1["condense (mS/cm)"],  color='skyblue', label='濃水電導度')
ax1.plot(data1.index, data1["discharge (mS/cm)"], color='gray', label='放流水電導度')
ax1.plot(data1.index, data1["fresh"], color='fuchsia', label='淡水電導度')
plt.xlabel('次')
plt.rcParams['font.family']='SimHei'
plt.ylabel('電導度')
plt.legend()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(data1.index,data1["rmrate"] , 'r')

===========================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%matplotlib inline
import plotly.figure_factory as FF

dataset = pd.read_csv("C:/Users/minghwu/Desktop/IHMStefanini_industrial_safety_and_health_database.csv")
dataset1=dataset.iloc[:,0:3]
dataset1=dataset1.dropna()
dataset1.columns  

=======================plotly============================
trace1 = go.Scatter(
    x=dataset1["time(mins)"],
    y=dataset1["conductivity(uS/cm)"],
    name='cond'
)
trace2 = go.Scatter(
    x=dataset1["time(mins)"],
    y=dataset1["current(A)"],
    name='I',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    title='Double Y Axis Example',
    yaxis=dict(
        title='cond'
    ),
    yaxis2=dict(
        title='yaxis2 title',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='overlaid histogram')
============================SNS===========================================

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
x=dataset1["time(mins)"]
y1=dataset1["conductivity(uS/cm)"]
y2=dataset1["current(A)"]
myfont = FontProperties(fname=r'E:\\ana\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\msjh.ttf')
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('電導度',fontproperties=myfont)
ax1.set_title("水質",fontproperties=myfont)

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, 'r')

ax2.set_ylabel('電流',fontproperties=myfont)

ax1.set_xlabel('時間',fontproperties=myfont)
plt.show()

![image](https://github.com/jasonfghx/python-imp-/blob/master/Capture.JPG)    
