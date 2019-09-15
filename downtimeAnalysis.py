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
#Associate a number with each event code (for plotting)
codes = df.code.unique()
codes.sort()
codeDict = dict(zip(codes, np.linspace(0,len(codes)-1,len(codes))))
df['codeIndex'] = df['code'].map(codeDict).astype(np.int16)

#Select only first run period
df = df.query('week<27')

def heatmapDowntime(df, xName, yName, xdim, ydim, durCeiling, yticklabels, mode, machine=None):
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
    f, axs = plt.subplots(1,1,figsize=(16,10))
    plt.title('%s\n '%machine, fontsize=20)
    if mode == 'counts':
        sns.heatmap(hmCounts, cmap='viridis', fmt='g', annot=True, linewidths=.5, cbar_kws={'label': 'Number of downtime events'}, yticklabels= yticklabels)
    elif mode == 'summed':
        sns.heatmap(hmDur, cmap='viridis', fmt='g', annot=True,linewidths=.5, cbar_kws={'label': 'Summed downtime (hours)'}, yticklabels= yticklabels, vmax=durCeiling) 
    plt.ylabel(xName, fontsize=16)
    plt.xlabel(yName, fontsize=16)
    plt.tight_layout()
    plt.show()


def histogramDowntime():
    #Histograms of machine downtime durations
    mList=['R1', 'R3', 'I']
    lStyle=['solid', 'dashed', 'dashdot']
    f, axs = plt.subplots(1,1,figsize=(16,10))
    plt.subplot(1,2,1)
    plt.hist(df.duration, bins=12, range=(0,120), linewidth=2, alpha=0.5, label='All machines')
    plt.title('Downtime duration (excluding events longer than 2 hours)')
    for i in range(0, len(mList)):
        plt.hist(df.query('machine==\'%s\''%mList[i]).duration, bins=12, range=(0,120), histtype='step', linewidth=2, linestyle=lStyle[i], label=mList[i])

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Counts')
    plt.legend()

    # #Histograms of machine downtime hour of day
    plt.subplot(1,2,2)
    plt.title('Downtime starting hour')
    plt.hist(df.hour, bins=24, range=(0,24), linewidth=2, alpha=0.5, label='All machines')
    for i in range(0, len(mList)):
        plt.hist(df.query('machine==\'%s\''%mList[i]).hour, bins=24, range=(0,24), histtype='step', linewidth=2, linestyle=lStyle[i], label=mList[i])
    plt.xlabel('Start hour')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.show()



def generatePlots():
    answer='y'
    while answer=='y' or answer=='Y':
        print('Plotting downtime')
        plottype = 0
        while ((plottype!=1) and (plottype!=2)):
            print(plottype)
            print('Generate histograms or heatmaps?\n1. Histograms\n2. Heatmaps:')
            plottype = int(input())

        if plottype==1:
            histogramDowntime()
        if plottype==2:
            machine = None
            while machine not in ['R1', 'R3', 'I', '']:
                print('Choose machine (R1 or R3 or I), Leave blank to show combined statistics:')
                machine = input()
            mode = None
            while mode not in ['counts', 'summed']:
                print('choose z-axis for heatmaps (counts or summed):')
                mode = input()
            #week number vs event code
            heatmapDowntime(df, 'codeIndex', 'week', 20, 27, durCeiling=300, machine=machine,  yticklabels=codes, mode=mode)
            #hour of day vs event code
            heatmapDowntime(df, 'codeIndex', 'hour', 20, 24, durCeiling=300, machine=machine,  yticklabels=codes, mode=mode)
            #downtime start hour vs week day
            heatmapDowntime(df, 'dayofweek', 'hour', 7, 24, durCeiling=300, machine=machine, yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], mode=mode)
            #Week number vs day of week
            heatmapDowntime(df, 'dayofweek', 'week', 7, 27, durCeiling=300, machine=machine, yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], mode=mode)
        print('make more plots(y/n)?')
        answer = input()

if __name__ == '__main__':
        generatePlots()



