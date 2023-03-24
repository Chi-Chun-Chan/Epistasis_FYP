import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#%% use this to define the plotting functions used to inspect inducer dependent epistasis
#set what colour you want to display the model in
observed_col = 'dimgray'
model_hill_all_col = 'springgreen'
model_therm_sens_col = 'slateblue'
#Figures function gives figures of standard deviation and mean epistasis for mutants grouped by inducer conc and node mutated
#Must input string in the form MODEL_SAMPLESTRAT and optionally a colour to label the graph
def Figures(model= 'model_hill_all', split_pt = False, model_col:str = model_hill_all_col):
    #give a more readable name to the model
    model_name = f"{model.split('_')[1]}, {model.split('_')[2]}"
    #function to get Eps for a given model
    def get_modelEps(model:str = 'observed'):
        columns = [ 'genotype category', 'inducer level', 'genotype','Ep']
        df = pd.read_excel(f"../results/Eps_{model}.xlsx")[columns]
        df_Eps = pd.DataFrame(columns[:-2]+['std', 'node']).set_index(0).T
        #get all instances of each mutant in its own dataframe and group to find the mean and std
        for node in ['Sensor', 'Regulator', 'Output']:
            for i in range(1,11):
                mut = f"{node[0]}{i}"
                search = df.loc[((df['genotype'].str.startswith(mut + '_')) | (df['genotype'].str.endswith('_' + mut))| (df['genotype'].str.contains('_' + mut + '_')))].groupby(['genotype category', 'inducer level']) 
                df_mut_mean = search.mean().rename(columns={"Ep": "mean"})
                df_mut_std = search.std().rename(columns={"Ep": "std"})
                df_mut = pd.concat([df_mut_mean,df_mut_std], axis = 1).reset_index()
                #need to reorder inducer level column for plotting later
                df_mut['order'] = [3,1,2]*(int(len(df_mut)/3))
                df_mut = df_mut.sort_values(by = "order").drop('order', axis = 1)
                #add a label for the node
                df_mut['node'] = [node]*len(df_mut)
                #add mutant df to df with all mutants in
                df_Eps = pd.concat([df_Eps, df_mut])
        #add a column indicating the model
        #give a more readable name to the models
        if model != 'observed':
            model = f"{model.split('_')[1]}, {model.split('_')[2]}"
        df_Eps['model'] = [model]*len(df_Eps)
        #prefer 'LMH' to 'inducer level' since only one word, same for genotype category
        return df_Eps.rename(columns = {'inducer level': 'LMH', 'genotype category': 'cat'}).reset_index(drop = True)

    df_Eps_obs = get_modelEps() #get obseved eps
    df_Eps_mod = get_modelEps(model)#get model data
    df_Eps = pd.concat([df_Eps_obs, df_Eps_mod])
    #stick node, and LMH columns together so that seaborn can group them properly
    df_Eps["desc"] = df_Eps["node"].map(str) + ", " + df_Eps["LMH"]

    #manually set the x tick positions and names
    x_ticks = []
    x_names = []
    for i in range(df_Eps["desc"].nunique()):
        adj = 0.25
        if i % 3 == 1:
            x_ticks.append(i)
        elif i % 3 == 2:
            x_ticks.append(i-adj)
        else:
            x_ticks.append(i+adj)
    x_names = list(df_Eps["LMH"].unique())*3
    sec_x_names = list(df_Eps["node"].unique())

    def Plotter(ax, data, y):
        #violin plots for each node/inducer conc combination
        palette ={"observed": observed_col, model_name: model_col}
        sns.violinplot(ax = ax, data=data, x='desc', y=y, hue="model", split=True, dodge=False, palette=palette)
        #ax.xaxis.set_ticks(x_ticks)
        ax.set_xticklabels(x_names, fontsize=axis_size)
        ax.set_xlabel('')
        #coloured backgrounds to make it clear which plot goes with which node
        ax.add_patch(patches.Rectangle((-0.5,-100), 9/3, 1000, alpha = 0.1, color = 'r', zorder = 0.1)) 
        ax.add_patch(patches.Rectangle((-0.5+9/3,-100), 9/3, 1000, alpha = 0.1, color = 'g', zorder = 0.1)) 
        ax.add_patch(patches.Rectangle((-0.5+(9*2)/3,-100), 9/3, 1000, alpha = 0.1, color = 'b', zorder = 0.1)) 

    def set_axs(ax1, ax2):
        ax1.axhline(zorder = 0.5, c = 'darkgrey', linestyle = '--')
        #axis Ticks
        ax1.set_xticks([])
        secax = ax2.secondary_xaxis('bottom')
        secax.set_xticks(x_ticks[1::3], sec_x_names, fontsize = axis_size)
        #axis labeled by inducer conc
        ax1.tick_params(axis = "y", labelsize = tick_size)
        ax2.tick_params(axis = "y", labelsize = tick_size)
        secax.tick_params(pad=pad)
        #ax.set_xticklabels( names )
        #axis labelled by node 
        for ticklabel, tickcolor in zip(secax.get_xticklabels(), ['r', 'g', 'b']):
            ticklabel.set_color(tickcolor)
        #axis labels
        ax1.set_ylabel(f"mean of $\epsilon$", fontsize = axis_size)
        ax2.set_ylabel(f"$\sigma$ of $\epsilon$", fontsize = axis_size)
        #legend
        ax1.get_legend().remove()
        ax2.get_legend().remove()

    if split_pt == False:
        title_size = 15
        axis_size = 14
        tick_size = 10
        pad = 20#title padding
        pad_legend = -0.1 #legend padding from bottom of x axis
        Fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8.27,7))
        Plotter(ax1, data = df_Eps, y= "mean")
        Plotter(ax2, data = df_Eps, y= "std")
        set_axs(ax1, ax2)
        #file path to save figure to
        file_path = f"../results/Ep_compare_split_{model}.jpg"

    else:
        #if separate plots for pairs and trips
        title_size = 50
        axis_size = 40
        tick_size = 25
        pad = 70 #title pad
        pad_legend = -0.14
        Fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (32,14))
        Plotter(ax1, data = df_Eps.loc[df_Eps.cat == "pairwise"], y= "mean")
        Plotter(ax3, data = df_Eps.loc[df_Eps.cat == "pairwise"], y= "std")
        Plotter(ax2, data = df_Eps.loc[df_Eps.cat == "triplet"], y= "mean")
        Plotter(ax4, data = df_Eps.loc[df_Eps.cat == "triplet"], y= "std")
        set_axs(ax1, ax3)
        set_axs(ax1 = ax2, ax2 = ax4)
        ax2.set(ylabel= '')
        ax2.set_title('Triplet', c = 'orange', fontsize = axis_size)
        ax1.set_title('Pairwise', c = 'purple', fontsize = axis_size)
        ax4.set(ylabel= '')
        ax3.set_xticklabels(['L', 'M', 'H']*3)
        ax4.set_xticklabels(['L', 'M', 'H']*3)
        #file path to be saved to later
        file_path = f"../results/Ep_compare_split_{model}_pt.jpg"
    #legend
    handles, labels = ax2.get_legend_handles_labels()
    Fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, pad_legend), fontsize = axis_size, edgecolor = 'w')
    Fig.suptitle(f"Mean and Varience of Calculated versus Obseved Epistasis for \n{model.split('_')[1]} model with sampling strategy '{model.split('_')[2]}'".title(), fontsize = title_size)
    Fig.tight_layout()
    #save to jpg
    Fig.savefig(file_path, bbox_inches='tight')
    return Fig 
#%%
observed = Figures(model = 'model_hill_all',split_pt = False, model_col = model_hill_all_col)
observed = Figures(model = 'model_hill_all',split_pt = True, model_col = model_hill_all_col)
observed = Figures(model = 'model_thermodynamic_sensor',split_pt = False, model_col = model_therm_sens_col)
observed = Figures(model = 'model_thermodynamic_sensor',split_pt = True, model_col = model_therm_sens_col)

#%%
#plot comparing epistasis at different inducer concs
#first, get a dataframe with Ep value at high, hedium and low I conc on one row
def indComp(model1:str='model_hill_all', model2:str = 'model_thermodynamic_sensor'):
    #read in all three dfs
    df_model1 = pd.read_excel('../results/Eps_'+str(model1)+'.xlsx', index_col=0).set_index('genotype')
    df_model2 = pd.read_excel('../results/Eps_'+str(model2)+'.xlsx', index_col=0).set_index('genotype')
    df_obs = pd.read_excel('../results/Eps_observed.xlsx', index_col=0).set_index('genotype')

    #reorganise data so that epistasis at low, medium and high I concs are easily compared
    def reorder(df):
        Ep_medium = pd.DataFrame(df['Ep'].loc[(df['inducer level']=='medium')]).rename(columns ={'Ep':'Ep_medium'})
        Ep_high = pd.DataFrame(df['Ep'].loc[(df['inducer level']=='high')]).rename(columns={'Ep':'Ep_high'})
        df = df[['Ep', 'genotype category']].loc[(df['inducer level']=='low')]
        df = df.rename(columns={'Ep':'Ep_low'})
        df = df.join(Ep_high.join(Ep_medium))
        return df

    df_model1 = reorder(df_model1)
    df_model2 = reorder(df_model2)
    df_obs = reorder(df_obs)

    #get proportions that are in each quadrant of the plot using the following dataframes
    def proportions(df):
        df_up = df[((df['Ep_low'] < df['Ep_medium']) & (df['Ep_medium'] < df['Ep_high']))]
        df_down = df[(df['Ep_low'] > df['Ep_medium']) & (df['Ep_medium'] > df['Ep_high'])]
        df_peak = df[(df['Ep_low'] < df['Ep_medium']) & (df['Ep_medium'] > df['Ep_high'])]
        df_trough = df[(df['Ep_low'] > df['Ep_medium']) & (df['Ep_medium'] < df['Ep_high'])]
        #totals:
        n,n_pair, n_triplet = len(df),len(df[df['genotype category']=='pairwise']), len(df[df['genotype category']=='triplet'])

        n_up = [len(df_up), len(df_up[df_up['genotype category']=='pairwise']), len(df_up[df_up['genotype category']=='triplet'])]

        n_down = [len(df_down), len(df_down[df_down['genotype category']=='pairwise']), len(df_down[df_down['genotype category']=='triplet'])]

        n_peak = [len(df_peak), len(df_peak[df_peak['genotype category']=='pairwise']), len(df_peak[df_peak['genotype category']=='triplet'])]
        n_trough = [len(df_trough), len(df_trough[df_trough['genotype category']=='pairwise']), len(df_trough[df_trough['genotype category']=='triplet'])]
        n_peak = [len(df_peak), len(df_peak[df_peak['genotype category']=='pairwise']), len(df_peak[df_peak['genotype category']=='triplet'])]
        return n,n_pair, n_triplet, n_up, n_down, n_peak, n_trough

    # % calculator
    def percent(x,n):
        return np.round(x*100/n,0)

    #now plot a scatter plot of med-low vs high-med
    def Plotter(ax, df):
        x = df.Ep_medium - df.Ep_low
        y= df.Ep_high - df.Ep_medium
        cols = np.where(df['genotype category'] == 'pairwise', 'purple', 'orange')
        ax.scatter(x, y, c = cols, zorder = 2.5, edgecolor = 'black', alpha = 0.65, label = ['pairwise', 'triple'])
        a = (df.Ep_medium - df.Ep_low).abs().max()
        b = (df.Ep_high - df.Ep_medium).abs().max()
        axis_lim = max(a,b)
        return axis_lim

    #define figure for scatter plots to go onto
    fig = plt.figure(figsize=(8, 9))
    shape = (100,80)
    ax_key1 = plt.subplot2grid(shape=shape, loc=(0, 20), rowspan = 20, colspan=40)
    ax_key2 = plt.subplot2grid(shape=shape, loc=(25, 45), rowspan = 35, colspan = 35)
    ax_obs = plt.subplot2grid(shape=shape, loc=(25, 5), rowspan = 35, colspan = 35)
    ax_1 = plt.subplot2grid(shape=shape, loc=(65,5), rowspan = 35, colspan = 35)
    ax_2 = plt.subplot2grid(shape=shape, loc=(65, 45), rowspan = 35, colspan = 35)

    a =Plotter(ax_obs, df_obs)
    b = Plotter(ax_1, df_model1)
    c = Plotter(ax_2, df_model2)
    axis_lim = max(a,b,c)

    def set_ax(ax, model:str, hidex:bool = False, hidey:bool = False):
        ax.axhline(y=0, c = 'darkgray', ls = '--')
        ax.axvline(x=0, c = 'darkgray', ls = '--')
        ax.set_xlim([-axis_lim*1.1,axis_lim*1.1])
        ax.set_ylim([-axis_lim*1.1,axis_lim*1.1])
        ax.set_xlabel('$\u03B5_{medium}$ - $\u03B5_{low}$')
        ax.set_ylabel('$\u03B5_{high}$ - $\u03B5_{medium}$')
        if model == 'observed':
            ax.set_title(f"{model}")
        else:
            ax.set_title(f"{model.split('_')[1]}, {model.split('_')[2]}")
        if hidex == True:
            ax.set_xticks([])
            ax.set_xlabel('')
        if hidey == True:
            ax.set_yticks([])
            ax.set_ylabel('')
    
    set_ax(ax_obs, 'observed', hidex=True)
    set_ax(ax_1, model1)
    set_ax(ax_2, model2, hidey=True)
    ax_key2.set(xticks = [], xlabel= '', ylabel = '', yticks = [], title = "key")
    ax_key1.set_axis_off()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Pairwise', markerfacecolor='purple', markeredgecolor = "k", markersize=13, alpha = 0.65), Line2D([0], [0], marker='o', color='w', label='Triple',  alpha = 0.65, markerfacecolor='orange',markeredgecolor = "k", markersize=13)]
    ax_1.legend(handles = legend_elements,  loc="upper left", bbox_to_anchor=(0, -0.15), edgecolor = 'w')
    #ax_obs.legend(frameon=False)

    #proportins mutants in each quadrant
    def writing(ax, n,n_pair, n_trip, n_up, n_down, n_peak, n_trough):
        def annotate(direction):
            adjx=  0.75; adjy = 0.9
            if direction == n_up:
                a, b, c = 1, 1, 0.9 #a & b position %s horizontally & vertically, c and d adjust pair/trip %s
            elif direction == n_down:
                a, b , c= -1,-1, 1.1
            elif direction == n_peak:
                a, b, c = 1,-1, 1.1
            else:
                a, b, c = -1,1 , 0.9
            x_pos, y_pos = a*axis_lim*adjx,b*axis_lim*adjy
            ax.text(x_pos,y_pos, str(percent(direction[0],n))+'%', verticalalignment="center", horizontalalignment='center', size = 20)
            ax.text(x_pos, y_pos*c, str(percent(direction[1],n_pair))+'%', verticalalignment="top", horizontalalignment='right', size = 9, c = 'purple')
            ax.text(x_pos, y_pos*c, str(percent(direction[2],n_trip))+'%', verticalalignment="top", horizontalalignment='left', size = 9, c = 'orange')
        annotate(n_up)
        annotate(n_down)
        annotate(n_peak)
        annotate(n_trough)
    
    writing(ax_obs, *proportions(df_obs))
    writing(ax_1, *proportions(df_model1))
    writing(ax_2, *proportions(df_model2))

    #plot a key for the inducer dependence plot
    x = np.array([1/6, 0.5, 5/6])
    y_trough = np.array([0.7, 0.3, 0.7])
    y_up = np.array([0.2, 0.5, 0.8])
    def plot(ax, x,y):
        return ax.plot(x,y, marker= 'o', markerfacecolor= 'lightgray',c= 'k', linestyle='-')
    plot(ax_key2,x,(-1)*y_trough)
    plot(ax_key2, (-1)*x, y_trough)
    plot(ax_key2, x, y_up)
    plot(ax_key2, -1*x, -1*np.flip(y_up))
    plot(ax_key1, x, y_up)
    plot(ax_key2, [-1/3, 0, 1/3], [0,0,0])
    ax_key2.axhline(zorder = 0.5, c = 'lightgrey', linestyle = '--')
    ax_key2.axvline(zorder = 0.5, c = 'lightgrey', linestyle = '--')
    ax_key1.text(0.499, 0.5, '{', verticalalignment='bottom', horizontalalignment='right', size = 35, c= 'k')
    ax_key1.text(0.37, 0.65, '$\epsilon_{high} - \epsilon_{medium}$', verticalalignment='center', horizontalalignment='right', size = 15, c= 'k')
    ax_key1.text(0.51, 0.5, '}', verticalalignment='top', horizontalalignment='left', size = 35, c= 'k')
    ax_key1.text(0.63, 0.32, '$\epsilon_{medium} - \epsilon_{low}$', verticalalignment='center', horizontalalignment='left', size = 15, c= 'k')
    ax_key2.text(0.4,0, 'inducer \nindependence', size = '7', verticalalignment = 'center')
    fig.suptitle('Inducer dependence of $\epsilon$', size = 25, y = 0.93)
    fig.tight_layout()
    fig.savefig(f"../results/Ep_indComp.jpg", bbox_inches='tight')
    return fig
#%%
indEps_observed = indComp(model1 = 'model_thermodynamic_sensor', model2 = "model_hill_all")

#%%
#plot significant epistasis
def Sig_Ep(model='observed'):
    df_M = pd.read_excel('../results/Eps_'+str(model)+'.xlsx', index_col=0)
    cols = np.where(df_M['Sig_Epistasis'] == True, 'darkgray', np.where(df_M['genotype category'] == 'pairwise', 'palegreen', 'lightskyblue'))
    alphas = np.where(df_M['Sig_Epistasis'] == True, 0.5, 0.8)
    fig, ax = plt.subplots()
    ax.scatter(df_M['Ep_pVal'], df_M['Ep'], c = cols,alpha = alphas, zorder = 2.5, edgecolor = 'black' )
    ax.axvline(0.05, c = 'darkgray', ls = '--')
    plt.show
    return fig
#%%

#Figures_scatter function gives figures of standard deviation and mean epistasis for mutants grouped by inducer conc and node mutated
def Figures_scatter(model= 'observed'):
    columns = ['genotype', 'Ep', 'genotype category', 'inducer level']
    df = pd.read_excel('../results/Eps_'+model+'.xlsx')[columns]
    columns.append('node')
    df_Eps = pd.DataFrame(columns).set_index(0).T
    node_order = ['Sensor', 'Regulator', 'Output']
    for nodes in node_order:
        for cat in ['pairwise', 'triple']:
            for ind in ['low', 'medium', 'high']:
                for i in range(1,11):
                    node = nodes[0] +str(i)
                    rows = df['Ep'].loc[((df['genotype'].str.startswith(node + '_')) | (df['genotype'].str.endswith('_' + node))| (df['genotype'].str.contains('_' + node + '_'))) & df['genotype category'].str.contains(cat) & df['inducer level'].str.contains(ind)]
                    mean = rows.mean()
                    std = rows.std()
                    df_node = pd.DataFrame({'genotype':[node], 'mean': [mean], 'std': [std], 'genotype category': [cat], 'inducer level': [ind], 'node':[nodes]}).set_index('genotype')
                    df_Eps = pd.concat([df_Eps, df_node])
    df_Eps = df_Eps.drop(columns = ['genotype', 'Ep']).rename(columns = {'inducer level': 'LMH'})

    df_means_p = df_Eps.loc[df_Eps['genotype category']== 'pairwise'].drop(columns='std').rename(columns = {'mean': 'Epistasis'})
    df_means_t = df_Eps.loc[df_Eps['genotype category']== 'triple'].drop(columns='std').rename(columns = {'mean': 'Epistasis'})
    df_vars_p = df_Eps.loc[df_Eps['genotype category']== 'pairwise'].drop(columns='mean').rename(columns = {'std': 'Epistasis'})
    df_vars_t = df_Eps.loc[df_Eps['genotype category']== 'triple'].drop(columns='mean').rename(columns = {'std': 'Epistasis'})

    group = ['node','LMH']
    col_choice = ['r', 'g', 'b']
    #group data by node and inducer cons ready to be plotted on a figure
    def Plot_setup(df:pd.DataFrame):
        grouped = df.groupby(group, sort=False)
        names, vals, xs, bar_pos, cols = [], [],[], [], []
        for i, (name, subdf) in enumerate(grouped):
            names.append(name[1])
            vals.append(subdf['Epistasis'].tolist())
            bar_pos += [i+1]
            xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))
            if i < 3:
                cols += [[col_choice[0]]*subdf.shape[0]]
            elif i < 6:
                cols += [[col_choice[1]]*subdf.shape[0]]
            else:
                cols += [[col_choice[2]]*subdf.shape[0]]
        #posttions for boxplots and x values for scatter plots
        for i in range(3):
            adj = 0.3
            xs[3*i] += adj
            bar_pos[3*i] += adj
            xs[3*i+2] -= adj
            bar_pos[3*i+2] -= adj
        return names, vals, xs, bar_pos, cols

    #defina a plotting function
    def Plotter(ax, names, vals, xs, bar_pos, cols,isMean = False, triple = False):
        #names = mean_P[0]
        #bar_pos = mean_P[3]
        if triple == True:
            title = 'triple'
            title_col = 'orange'
        else:
            title = 'pairwise'
            title_col = 'purple'
        ax.set_title(title, c = title_col) 
        for x, val, col in zip(xs, vals, cols):
            ax.scatter(x, val, alpha=0.4, c = col)
            ax.errorbar(np.mean(x) , np.mean(val), np.std(val), linestyle='None', fmt='_', c = 'k', elinewidth = 1.5, capsize = 1.5)
        if isMean == True:
            ax.axhline(zorder = 0.5, c = 'lightgrey', linestyle = '--')
            #ax.set_ylim(min(df_Eps['mean'])*1.1, max(df_Eps['mean'])*1.1) 
            ax.set_ylim(-0.7, 1.3) #hard code limits to easily compare between models
        else:
            #ax.set_ylim(0, max(df_Eps['std'])*1.1)
            ax.set_ylim(0, 0.8) 
        ax.xaxis.set_ticks(bar_pos)
        ax.set_xticklabels(names,rotation=45, fontsize=8)
        secax = ax.secondary_xaxis('bottom')
        secax.set_xticks(bar_pos[1::3], node_order)
        secax.tick_params(pad=40)
        #ax.set_xticklabels( names )
        for ticklabel, tickcolor in zip(secax.get_xticklabels(), col_choice):
            ticklabel.set_color(tickcolor)
        return ax

    #Define the figures, mean first,
    Fig_obs_Mean, (means_p, means_t) = plt.subplots(1,2)
    mean_P = Plot_setup(df_means_p)
    Plotter(means_p, *mean_P, True)
    mean_T = Plot_setup(df_means_t)
    Plotter(means_t,*mean_T, True, triple = True)
    means_p.set_ylabel('mean $\epsilon$')
    Fig_obs_Mean.suptitle(str(model))
    plt.show()
    Fig_obs_Mean.savefig("../results/"+model+"_EpMean.jpg")

    #then varience
    Fig_obs_Var, (vars_p, vars_t) = plt.subplots(1,2)
    var_P = Plot_setup(df_vars_p)
    Plotter(vars_p, *var_P)
    var_T = Plot_setup(df_vars_t)
    Plotter(vars_t,*var_T, triple = True)
    vars_p.set_ylabel('standard deviation of $\epsilon$')
    Fig_obs_Var.suptitle(str(model))
    plt.show()
    Fig_obs_Var.savefig("../results/"+model+"EpStd.jpg")
    return Fig_obs_Mean, Fig_obs_Var

#now to get totals and proportions for more specific plots
#lmh indicates Ep < 0 at low, medium and high I conc
def Above_below_zero():
    up_lmh = df_up[(df_up.Ep_low < 0) & ( df_up.Ep_medium < 0) & (df_up.Ep_high < 0)] 
    up_lm = df_up[(df_up.Ep_low < 0) & ( df_up.Ep_medium < 0) & (df_up.Ep_high > 0)] 
    up_l = df_up[(df_up.Ep_low < 0) & ( df_up.Ep_medium > 0) & (df_up.Ep_high > 0)] 
    up_over = df_up[(df_up.Ep_low > 0) & ( df_up.Ep_medium > 0) & (df_up.Ep_high > 0)] 

    peak_lmh = df_peak[(df_peak.Ep_low < 0) & ( df_peak.Ep_medium < 0) & (df_peak.Ep_high < 0)] 
    peak_lm = df_peak[(df_peak.Ep_low < 0) & ( df_peak.Ep_medium < 0) & (df_peak.Ep_high > 0)] 
    peak_l = df_peak[(df_peak.Ep_low < 0) & ( df_peak.Ep_medium > 0) & (df_peak.Ep_high > 0)] 
    peak_over = df_peak[(df_peak.Ep_low > 0) & ( df_peak.Ep_medium > 0) & (df_peak.Ep_high > 0)] 

    trough_lmh = df_trough[(df_trough.Ep_low < 0) & ( df_trough.Ep_medium < 0) & (df_trough.Ep_high < 0)] 
    trough_lm = df_trough[(df_trough.Ep_low < 0) & ( df_trough.Ep_medium < 0) & (df_trough.Ep_high > 0)] 
    trough_l = df_trough[(df_trough.Ep_low < 0) & ( df_trough.Ep_medium > 0) & (df_trough.Ep_high > 0)] 
    trough_over = df_trough[(df_trough.Ep_low > 0) & ( df_trough.Ep_medium > 0) & (df_trough.Ep_high > 0)] 

    down_lmh = df_down[(df_down.Ep_low < 0) & ( df_down.Ep_medium < 0) & (df_down.Ep_high < 0)] 
    down_lm = df_down[(df_down.Ep_low < 0) & ( df_down.Ep_medium < 0) & (df_down.Ep_high > 0)] 
    down_l = df_down[(df_down.Ep_low < 0) & ( df_down.Ep_medium > 0) & (df_down.Ep_high > 0)] 
    down_over = df_down[(df_down.Ep_low > 0) & ( df_down.Ep_medium > 0) & (df_down.Ep_high > 0)] 
#%%