#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# equilibration time (from correlation_time.py)
equil_jR2R3 = 110 #ns
equil_jR2R3_P301L = 80  #ns
equil_jR1R3 = 140 #ns
equil_list = [equil_jR2R3, equil_jR2R3_P301L, equil_jR1R3]

# load the number of intermolecular hydrogen bond per frame (created with  `gmx hbond ...` with mainchain+H selected for each monomer)
data_jR2R3_hbonds = np.loadtxt('jR2R3_hbnum.xvg',comments=['@','#'])[equil_jR2R3*100:,1]
data_jR2R3_P301L_hbonds = np.loadtxt('jR2R3_P301L_hbnum.xvg',comments=['@','#'])[equil_jR2R3_P301L*100:,1]
# data_P301S_hbonds = np.loadtxt('P301S_hbnum.xvg',comments=['@','#'])[equil_P301S*100:,1]
data_jR1R3_hbonds = np.loadtxt('jR1R3_hbnum.xvg',comments=['@','#'])[equil_jR1R3*100:,1]

# num independent (from correlation_time.py)
indep_jR2R3 = 1274 
indep_jR2R3_P301L = 1139
indep_jR1R3 = 1353

#%% organize masks for filtering
peptides = ['jR2R3','jR2R3_P301L','jR1R3']
monomers = ['A','B']

monomer_dfs = {}  # Initialize as a dictionary

for peptide in peptides:
    monomer_dfs[peptide] = {}  # Initialize each peptide key to hold another dictionary
    for monomer in monomers:
        # csv file from hbond_masks.py
        df = pd.read_csv(f'hbond_masks_{peptide}_dimer_monom{monomer}.csv',index_col=0) 
        df = df.astype(bool)
        cluster1 = np.logical_and(df['r300-r305'],df['Q307-K298'])
        cluster2 = np.logical_and(df['V309-K298'],df['r300-Q307'])
        # stack the two clusters
        df_clusters = pd.concat([cluster1,cluster2],axis=1)
        df_clusters.columns = ['cluster1', 'cluster2']   # Assign new column names
        monomer_dfs[peptide][monomer] = df_clusters

# # Now you can access the DataFrames like this:
# print(monomer_dfs['jR2R3']['A'])
# print(monomer_dfs['jR1R3']['B'])
#%% Determine frames where at least one monomer is cluster 1;
### where at least one monomer is cluster 2; and where neither monomer is cluster 1 or 2
populations_list = []
counts_list = []

for i,peptide in enumerate(peptides):
    cluster1 = np.logical_or(monomer_dfs[peptide]['A']['cluster1'][int(equil_list[i]*100):],monomer_dfs[peptide]['B']['cluster1'][int(equil_list[i]*100):])
    cluster2 = np.logical_or(monomer_dfs[peptide]['A']['cluster2'][int(equil_list[i]*100):],monomer_dfs[peptide]['B']['cluster2'][int(equil_list[i]*100):])
    combine_clusters = np.zeros(len(cluster1))  # Initialize the combined clusters as an array of zeros
    # (0 means neither cluster 1 or cluster 2)
    combine_clusters[cluster1] = 1  # Assign 1 to elements in cluster 1
    combine_clusters[cluster2] = 2  # Assign 2 to elements in cluster 2

    unique, counts = np.unique(combine_clusters, return_counts=True)
    populations_list.append(combine_clusters)
    counts_list.append(counts)

# turn populations_list into a dictionary
populations_dict = {peptide: population for peptide, population in zip(peptides, populations_list)}
# access like this:
# populations_dict['jR2R3']

counts_df = pd.DataFrame(counts_list,index=peptides,columns=['neither is cluster 1 or 2','at least one is cluster1','at least one is cluster2'])
# reorder columns so 'neither is cluster1/2' is last
counts_df = counts_df[['at least one is cluster1','at least one is cluster2','neither is cluster 1 or 2']]

# divide by the sum of each row
counts_df = counts_df.div(counts_df.sum(axis=1), axis=0)*100

#%% jR2R3: UNFILTERED, 1or2 clusterA, 1or2 clusterB, or neither clusterA/B
bootstrap_num = 500
data_bootstrapped = np.zeros((4,bootstrap_num)) # unfiltered, cluster1, cluster2, neither
bins = [0,5,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_hbonds),indep_jR2R3)
    boot_hbonds = data_jR2R3_hbonds[indices]
    data_bootstrapped[0,k] = np.histogram(boot_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

    booted_filter = populations_dict['jR2R3'][indices]

    cluster1_mask = booted_filter == 1
    cluster2_mask = booted_filter == 2
    neither_mask  = booted_filter == 0

    cluster1_hbonds = boot_hbonds[cluster1_mask]
    cluster2_hbonds = boot_hbonds[cluster2_mask]
    neither_hbonds  = boot_hbonds[neither_mask]

    data_bootstrapped[1,k] = np.histogram(cluster1_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[2,k] = np.histogram(cluster2_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[3,k] = np.histogram(neither_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

hbond_frac_confidence = np.zeros((4,3)) # unfiltered, cluster1, cluster2, neither
confidence = 0.90
percentiles = [5, 50, 95] # the percentiles we're interested in

# Loop over each row
for i in range(data_bootstrapped.shape[0]):
    sorted_row = np.sort(data_bootstrapped[i])
    for j, percentile in enumerate(percentiles):
        hbond_frac_confidence[i, j] = np.percentile(sorted_row, percentile)

# Create pandas DataFrame
cluster_names = ['Unfiltered', 'Cluster1', 'Cluster2', 'Neither']
percentile_names = [f'{percentile}%' for percentile in percentiles]
jR2R3_hbond_confidence = pd.DataFrame(hbond_frac_confidence, index=cluster_names, columns=percentile_names)

#%% jR2R3_P301L: UNFILTERED, 1or2 clusterA, 1or2 clusterB, or neither clusterA/B
bootstrap_num = 500
data_bootstrapped = np.zeros((4,bootstrap_num)) # unfiltered, cluster1, cluster2, neither
bins = [0,5,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR2R3_P301L_hbonds),indep_jR2R3_P301L)
    boot_hbonds = data_jR2R3_P301L_hbonds[indices]
    data_bootstrapped[0,k] = np.histogram(boot_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

    booted_filter = populations_dict['jR2R3_P301L'][indices]

    cluster1_mask = booted_filter == 1
    cluster2_mask = booted_filter == 2
    neither_mask  = booted_filter == 0

    cluster1_hbonds = boot_hbonds[cluster1_mask]
    cluster2_hbonds = boot_hbonds[cluster2_mask]
    neither_hbonds  = boot_hbonds[neither_mask]

    data_bootstrapped[1,k] = np.histogram(cluster1_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[2,k] = np.histogram(cluster2_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[3,k] = np.histogram(neither_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

hbond_frac_confidence = np.zeros((4,3)) # unfiltered, cluster1, cluster2, neither
confidence = 0.90
percentiles = [5, 50, 95] # the percentiles we're interested in

# Loop over each row
for i in range(data_bootstrapped.shape[0]):
    sorted_row = np.sort(data_bootstrapped[i])
    for j, percentile in enumerate(percentiles):
        hbond_frac_confidence[i, j] = np.percentile(sorted_row, percentile)

# Create pandas DataFrame
cluster_names = ['Unfiltered', 'Cluster1', 'Cluster2', 'Neither']
percentile_names = [f'{percentile}%' for percentile in percentiles]
jR2R3_P301L_hbond_confidence = pd.DataFrame(hbond_frac_confidence, index=cluster_names, columns=percentile_names)

#%%
#%% jR1R3: UNFILTERED, 1or2 clusterA, 1or2 clusterB, or neither clusterA/B
bootstrap_num = 500
data_bootstrapped = np.zeros((4,bootstrap_num)) # unfiltered, cluster1, cluster2, neither
bins = [0,5,np.inf]

for k in range(bootstrap_num):
    indices = np.random.choice(len(data_jR1R3_hbonds),indep_jR1R3)
    boot_hbonds = data_jR1R3_hbonds[indices]
    data_bootstrapped[0,k] = np.histogram(boot_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

    booted_filter = populations_dict['jR1R3'][indices]

    cluster1_mask = booted_filter == 1
    cluster2_mask = booted_filter == 2
    neither_mask  = booted_filter == 0

    cluster1_hbonds = boot_hbonds[cluster1_mask]
    cluster2_hbonds = boot_hbonds[cluster2_mask]
    neither_hbonds  = boot_hbonds[neither_mask]

    data_bootstrapped[1,k] = np.histogram(cluster1_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[2,k] = np.histogram(cluster2_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)
    data_bootstrapped[3,k] = np.histogram(neither_hbonds,bins=bins,
                                        density=False)[0][1]/len(indices)

hbond_frac_confidence = np.zeros((4,3)) # unfiltered, cluster1, cluster2, neither
confidence = 0.90
percentiles = [5, 50, 95] # the percentiles we're interested in

# Loop over each row
for i in range(data_bootstrapped.shape[0]):
    sorted_row = np.sort(data_bootstrapped[i])
    for j, percentile in enumerate(percentiles):
        hbond_frac_confidence[i, j] = np.percentile(sorted_row, percentile)

# Create pandas DataFrame
cluster_names = ['Unfiltered', 'Cluster1', 'Cluster2', 'Neither']
percentile_names = [f'{percentile}%' for percentile in percentiles]
jR1R3_hbond_confidence = pd.DataFrame(hbond_frac_confidence, index=cluster_names, columns=percentile_names)

#%%
# Define data
dataframes = [jR2R3_hbond_confidence, jR2R3_P301L_hbond_confidence, jR1R3_hbond_confidence]
labels = ['jR2R3', 'jR2R3_P301L', 'jR1R3']
clusters = ['Unfiltered', 'Cluster1', 'Cluster2']
colors = [(170/255,68/255,153/255), (17/255,119/255,51/266), (51/255,34/255,136/266)]

dataframes = [jR1R3_hbond_confidence, jR2R3_hbond_confidence, jR2R3_P301L_hbond_confidence]
labels = ['jR1R3', 'jR2R3', 'jR2R3_P301L']
clusters = ['Unfiltered', 'Cluster1', 'Cluster2']
colors = [(51/255,34/255,136/266), (170/255,68/255,153/255), (17/255,119/255,51/266)]
colors = [(51/255,34/255,136/266), (170/255,68/255,153/255), (136/255,204/255,238/255)]


# Calculate bar positions
num_clusters = len(clusters)
num_labels = len(labels)
bar_width = 0.2  # adjust as needed
cluster_gap = 0.2  # adjust as needed
bar_positions = np.arange(num_clusters) * (num_labels * bar_width + cluster_gap)

# Create plot
fig, ax = plt.subplots()

for i, df in enumerate(dataframes):
    for j, cluster in enumerate(clusters):
        position = bar_positions[j] + i * bar_width
        mean = df.loc[cluster, '50%']*100
        low = df.loc[cluster, '5%']*100
        high = df.loc[cluster, '95%']*100
        ax.bar(position, mean, width=bar_width, color=colors[i])
        ax.errorbar(position, mean, yerr=[[mean-low], [high-mean]], fmt='none', color='black')

# Format plot
ax.set_xticks(bar_positions + num_labels * bar_width / 2 - 0.1)
ax.set_xticklabels(['full dimer\nensemble','Cluster 1\npopulation',
                    'Cluster 2\npopulation'],fontsize=15)
ax.set_ylabel('% Probability of 5+\nBackbone Inter H-Bonds',fontsize=15)  # replace with appropriate label


import matplotlib.patches as mpatches
patch1 = mpatches.Patch(color=colors[1], label='jR2R3')
patch2 = mpatches.Patch(color=colors[2], label='jR2R3_P301L')
patch3 = mpatches.Patch(color=colors[0], label='jR1R3')
plt.legend(handles=[patch3, patch1, patch2],fontsize=15)

plt.tight_layout()

# save figure
plt.savefig('dimer_inter_hbonds_cluster_populations.png',dpi=300)

plt.show()


# %%

