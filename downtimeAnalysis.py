# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from matplotlib import gridspec
import matplotlib.ticker as ticker

###########################
#LOADING AND PREPROCESSING#
###########################
df=pd.read_csv('downtime.csv', usecols=[1, 2, 4, 5, 6], names=['date','time', 'code', 'machine', 'duration'], skiprows=[0])
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

#Schedule
sch=pd.read_csv('schedule.csv', usecols=[0,1, 2, 3, 4, 5], names=['date','week', 'year', 'r1Plan', 'r3Plan', 'spfPlan'], skiprows=[0])
sumR1 = sch.sum().r1Plan
sumR3 = sch.sum().r3Plan
sumI = sch.sum().spfPlan
delivDict = {'R1': sumR1, 'R3': sumR3, 'I': sumI}
for index, row in df.iterrows():
    df.at[index, 'percOfPlan'] = 100 * row['duration'] / (60*delivDict[row['machine']])
    df.at[index, 'percOfDown'] = 100 * row['duration'] / df.groupby('machine').sum().duration[row['machine']]

def getColor(c, N, idx):
    """Generate colormap for plots.
    From https://stackoverflow.com/questions/45612129/cdf-matplotlib-not-enough-colors-for-plot-python
    Args: cmap=colormap, N=number of discrete extracted colors, idx=index of color to extract"""
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

def heatmapDowntime(df, xName, yName, xdim, ydim, durCeiling, yticklabels, mode, xLab, yLab, machine=None):
    if not machine:
        machine = 'All machines'
    else:
        df = df.query('machine==\'%s\''%machine)
    hm = np.zeros((xdim, ydim))
    for index, row in df.iterrows():
        h = row[xName]
        d = row[yName]
        if mode == "counts":
            hm[h][d] += 1
            Z="Counts"
        elif mode == "summed":
            hm[h][d] += row.duration
            Z="Summed\ndowntime duration"
        elif mode == "percOfPlan":
            hm[h][d] += row.percOfPlan
            Z="\% of planned\ndelivery"
        elif mode == "percOfDown":
            hm[h][d] += row.percOfDown
            Z="\% of total\ndowntime"

    f, axs = plt.subplots(1,1,figsize=(16,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5])
    gs.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
    ax = plt.subplot(gs[1,0])

    axRight = plt.subplot(gs[1,1], sharey=ax)
    plt.setp(axRight.get_yticklabels(), visible=False)
    axTop = plt.subplot(gs[0,0], sharex=ax)
    plt.setp(axTop.get_xticklabels(), visible=False)
    
    axCorner = plt.subplot(gs[0,1], sharex=axRight, sharey=axTop)
    axCorner.text(0, 2, "{0}\n\nZ-axis: {1}".format(machine, Z), fontsize=16)
    plt.setp(axCorner.get_yticklabels(), visible=False)
    plt.setp(axCorner.get_xticklabels(), visible=False)


    axRight = plt.subplot(gs[1,1], sharey=ax)
    axTop = plt.subplot(gs[0,0], sharex=ax)
    arr1 = np.append(np.sum(hm,1), 0)
    axRight.fill_betweenx(x1=arr1, x2=0, y=np.linspace(0, len(arr1)-1, len(arr1)),  alpha=0.5, step="post", color=getColor('viridis',10,4))
    arr2 = np.append(np.sum(hm,0), 0)
    axTop.fill_between(y1=arr2, x=np.linspace(0, len(arr2)-1, len(arr2)), y2=0, alpha=0.5, step="post", color=getColor('viridis',10,4))

    plt.axes(ax)


    if mode == 'counts':
        sns.heatmap(hm, cmap='viridis', fmt='g', annot=True, linewidths=.5, cbar_kws={'label': 'Number of downtime events'}, yticklabels= yticklabels, cbar=False)
    elif mode == 'summed':
        sns.heatmap(hm, cmap='viridis', fmt='g', annot=True,linewidths=.5, cbar_kws={'label': 'Summed downtime (minutes)'}, yticklabels= yticklabels, vmax=durCeiling, cbar=False)
    elif mode == 'percOfPlan':
        sns.heatmap(hm, cmap='viridis', fmt='.1f', annot=True,linewidths=.5, cbar_kws={'label': 'Downtime (\% of total  planned delivery)'}, yticklabels= yticklabels, cbar=False)
    elif mode == 'percOfDown':
        sns.heatmap(hm, cmap='viridis', fmt='.1f', annot=True,linewidths=.5, cbar_kws={'label': 'Downtime (\% of total downtime)'}, yticklabels= yticklabels, cbar=False)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.ylabel(yLab, fontsize=16)
    plt.xlabel(xLab, fontsize=16)
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

def scatterDowntime(m):
    if m != '':
        code=list(df.query("machine=='%s'"%m).groupby('code').count().index)
        summed=list(df.query("machine=='%s'"%m).groupby('code').sum().duration)
        counts=list(df.query("machine=='%s'"%m).groupby('code').count().duration)
        plt.title('Machine: '+m, fontsize=20)
    else:
        code=list(df.groupby('code').count().index)
        summed=list(df.groupby('code').sum().duration)
        counts=list(df.groupby('code').count().duration)
        plt.title('All machines', fontsize=20)
    D=pd.DataFrame([code, summed, counts]).transpose()
    D.columns=['code', 'durSum', 'counts']
    for i, row in D.iterrows():
        if D.counts[i]>=0:
            if m != '':
                D.at[i, 'stdDuration'] = np.std(df.query("machine=='%s' and code=='%s'"%(m, row['code'])).duration)
            else:
                D.at[i, 'stdDuration'] = np.std(df.query("code=='%s'"%(row['code'])).duration)
    for index, row in D.iterrows():
        if index%2 == 0:
            marker='o'
            ls='--'
        else:
            marker='v'
            ls='-'
        eb = plt.errorbar(row.counts, row.durSum/row.counts, row.stdDuration, label=row.code, color=getColor('gist_rainbow',len(D)+1,index), marker=marker, ms=10)
        eb[-1][0].set_linestyle(ls)
    plt.legend(fontsize=14)
    plt.xlabel('Number of events', fontsize=16)
    plt.ylabel('Average downtime (minutes)', fontsize=16)
    #plt.title(m)
    plt.xlim(0,30)
    plt.ylim(-20,280)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("plottype", type=str,
                        help="Generate heatmaps 'hm' or histograms 'hi' or scatterplots 'sc'",
                        choices=["hi", "hm", "sc"])
    parser.add_argument("--machine", type=str,
                        help="Use only data relating to 'R1' or 'R3' or 'I' or 'All'  machines",
                        choices=['R1', 'R3', 'I', 'All'], default="All")
    parser.add_argument("--zAxis", type=str,
                        help="What to plot on z-axis of a heatmap.",
                        choices=['counts', 'summed', 'percOfDown', 'percOfPlan'], default="counts")
    args = parser.parse_args()
    plottype = args.plottype
    if args.machine == "All":
        machine = ""
    else:
        machine = args.machine
    mode = args.zAxis

    if plottype == 'hi':
        histogramDowntime()
    elif plottype == 'sc':
        scatterDowntime(machine)
    elif plottype == 'hm':
        #week number vs event code
        heatmapDowntime(df, 'codeIndex', 'week', 20, 27, durCeiling=300, machine=machine,  yticklabels=codes, mode=mode, xLab="week", yLab="Downtime code")
        #hour of day vs event code
        heatmapDowntime(df, 'codeIndex', 'hour', 20, 24, durCeiling=300, machine=machine,  yticklabels=codes, mode=mode, xLab="Starting hour", yLab="Downtime code")
        #downtime start hour vs week day
        heatmapDowntime(df, 'dayofweek', 'hour', 7, 24, durCeiling=300, machine=machine, yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], mode=mode, xLab="Starting hour", yLab="Day")
        #Week number vs day of week
        heatmapDowntime(df, 'dayofweek', 'week', 7, 27, durCeiling=300, machine=machine, yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], mode=mode, xLab="Week", yLab="Day")
        

