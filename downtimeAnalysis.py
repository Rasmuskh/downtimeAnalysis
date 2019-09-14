# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv('DowntimeData.csv', usecols=[1, 2, 4, 5, 6], names=['date','time', 'code', 'machine', 'duration'], skiprows=[0])
#convert date from string to date type
df['date'] = df['date'].astype('datetime64[ns]')
#add columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.week
df['dayofweek'] = df['date'].dt.dayofweek
df['time'] = pd.to_datetime(df['time'],format= '%H:%M' ).dt.time
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour

#Select only first run period
df = df.query('week<27')

def mapDowntime(df, xName, yName, xdim, ydim, durCeiling, machine=None):
    if not machine:
        machine = 'All machines'
    else:
        df = df.query('machine==\'%s\''%machine)
    hmCounts = np.zeros((xdim, ydim))
    hmDur = np.zeros((xdim, ydim))
    for index, row in df.iterrows():
        h = row[xName]
        d = row[yName]   
        hmCounts[h][d] += 1
        hmDur[h][d] += row.duration
    f, axs = plt.subplots(2,1,figsize=(12,8))
    plt.subplot(2,1,1)
    plt.title('Machine: %s\n '%machine, fontsize=20)
    sns.heatmap(hmCounts, cmap='viridis', fmt='g', annot=True,linewidths=.5, cbar_kws={'label': 'Number of downtime events'})
    plt.ylabel(xName, fontsize=16)
    plt.subplot(2,1,2)
    sns.heatmap(hmDur, cmap='viridis', fmt='g', annot=True,linewidths=.5, cbar_kws={'label': 'Summed downtime'}, vmax=durCeiling) 
    plt.ylabel(xName, fontsize=16)  
    plt.xlabel(yName, fontsize=16)
    plt.tight_layout()
    plt.show()

# for machine in [None, 'R1', 'R3', 'I']:
#     mapDowntime(df, 'dayofweek', 'hour', 7, 24, durCeiling=300, machine=machine)
#     mapDowntime(df, 'dayofweek', 'week', 7, 27, durCeiling=300, machine=machine)

#make a similar plot for event codes using groupby
f, axs = plt.subplots(1,1,figsize=(12,8))
plt.title('Event codes vs week number', fontsize=20)
df2=df.groupby(['code', 'week']).sum().reset_index()
sns.heatmap(df2.set_index(['week', 'code']).duration.unstack(0), cmap='viridis', fmt='g', annot=True,linewidths=.5, vmin=0, vmax=300, cbar_kws={'label': 'Summed downtime'})
plt.show()
f, axs = plt.subplots(1,1,figsize=(12,8))
plt.title('Event codes vs week number', fontsize=20)
df2=df.groupby(['code', 'week']).count().reset_index()
sns.heatmap(df2.set_index(['week', 'code']).duration.unstack(0), cmap='viridis', fmt='g', annot=True,linewidths=.5, cbar_kws={'label': 'Number of downtime events'})
plt.show()

#Histograms of machine downtime durations
mList=['R1', 'R3', 'I']
lStyle=['solid', 'dashed', 'dashdot']
for i in range(0, len(mList)):
    plt.hist(df.query('machine==\'%s\''%mList[i]).duration, bins=20, range=(0,200), histtype='step', linewidth=2, linestyle=lStyle[i], label=mList[i])
plt.xlabel('Duration (minutes)')
plt.ylabel('Counts')
plt.legend()
plt.show()


#Histograms of machine downtime hour of day
mList=['R1', 'R3', 'I']
lStyle=['solid', 'dashed', 'dashdot']
for i in range(0, len(mList)):
    plt.hist(df.query('machine==\'%s\''%mList[i]).hour, bins=24, range=(0,24), histtype='step', linewidth=2, linestyle=lStyle[i], label=mList[i])
plt.xlabel('Start hour')
plt.ylabel('Counts')
plt.legend()
plt.show()
