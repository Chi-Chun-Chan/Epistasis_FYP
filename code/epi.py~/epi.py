import numpy as np 
import seaborn as sns
import sympy as sym
from os import listdir
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

## Initizalization of dummy variables (parameters of the model) for sympy
symbolnames = ['aR','aS','aO','cS','cR','cO','bR','bS','bO','nR','nS','nO','I']
basepars = ['A','B','C','N'] # nbame of the pars for each mutant
myVars = vars()
for symbolname in symbolnames:
    myVars[symbolname] = sym.symbols(symbolname)
sym.init_printing(use_unicode=True)
# variables can be accessed by their name (without the string commas), or using the string as myVars[string]





def stst(pars):
    ''' 
    returns steady state values of the ODE given a parameter set
    for the three nodes S,R,O. It also returns Oh which is the
    expected output fluorescence in the absence of regulator
    '''


    affS = np.power(pars['C_s']*pars['I'],pars['N_s'])
    Seq = (pars['A_s'] + pars['B_s']*affS)/(1+affS)

    affR = np.power(pars['C_r']*Seq,pars['N_r'])
    Req = pars['A_r'] + pars['B_r']/(1+affR)

    affO = np.power(pars['C_o']*(Req+Seq),pars['N_o'])
    Oeq =  (pars['A_o'] + pars['B_o']/(1+affO))*pars['F_o'] #multiply by F_o

    affOh = np.power(pars['C_o']*Req,pars['N_o']) #halfoutput is only sensor
    Oheq = pars['A_o'] + pars['B_o']/(1+affOh)

    return Seq,Req,Oeq,Oheq

def CompareEpistasis():


### This plot eventually needs to create a histogram with 4 curves: 1) Measured Epistasis 2) Predicted Epistasis full model 3) Linear epistasis 4) Quadratic epistasis

#epi_lab = # load from files

#### 1) Measured epistasis - TODO

### 2) Epistasis from full model. Use parameters from WT and single fittings to predict duplets and then calculate epistasis

    depth = 10000 # MAX = 10000, number of parameter sets to use per file
    folder = '../../results/Combined_mutant_params/Pairwise_params/'
    epi_model = pd.DataFrame()

    letters = r'[a-zA-Z]'

    for file in listdir(folder):

        # Extract from file title, which is the duplet
        mutant_letters = re.findall(letters, file)
        m1_str = mutant_letters[0].lower() # first character of filename string
        m2_str = mutant_letters[1].lower() # first character of filename string
        print('mutant_combo: {} with mutants {} and {}'.format(file,m1_str,m2_str))

        # Calulate WT fluorescence
        data_WT = pd.read_csv(folder+file)
        data_WT = data_WT.head(depth) # if depth<10000 then only a subset of parameters is loaded
        data_WT['I'] = 0.0002 # WT peak position
        WT_fluo = stst(data_WT)[2]

        # Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
        data_mut1 = data_WT.copy()
        data_duplet = data_WT.copy()
        data_mut2 = data_WT.copy()

        # Mutant 1 and duplet part
        for par in basepars: # for each parameter of a node
            data_mut1[par+'_'+m1_str] = data_mut1[par+'_'+m1_str]*data_mut1['M'+par+'_'+m1_str]
            data_duplet[par+'_'+m1_str] = data_duplet[par+'_'+m1_str]*data_duplet['M'+par+'_'+m1_str]
        if m1_str == 'o': # in the case that 1 of the mutatns is the outout, apply fluorescence correction. Note that this should not affect epistasis
            data_mut1['F_o'] = data_mut1['F_o']*data_mut1['MF_o']
            data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

        m1_fluo = stst(data_mut1)[2]

        # Mutant 2 and duplet part
        for par in basepars:
            data_mut2[par+'_'+m2_str] = data_mut2[par+'_'+m2_str]*data_mut2['M'+par+'_'+m2_str]
            data_duplet[par+'_'+m2_str] = data_duplet[par+'_'+m2_str]*data_duplet['M'+par+'_'+m2_str]
        if m2_str == 'o':
            data_mut2['F_o'] = data_mut2['F_o']*data_mut2['MF_o']
            data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

        m2_fluo = stst(data_mut2)[2] 

        # Duplet
        duplet_fluo = stst(data_duplet)[2]

        logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
        logG_model =  np.log10(duplet_fluo/WT_fluo)
        epi_model = pd.concat([epi_model,logG_model - logG_expected])
    
    plt.xlim([-0.5,0.5])    
    plt.hist(epi_model,bins = 100, range=[-0.5,0.5])
    plt.savefig('epi_model.pdf')
    plt.show()

#epi_lin = ## call second_derivative_matrix_loglog and singlets
#epi_quad = ## call second_derivative_matrix_loglog_var and singlets
    return epi_model


def second_derivative_matrix():

    # Output Fluorescence matrix
    affS = np.power(cS*I,nS)
    Seq = (aS + bS*affS)/(1+affS)
    affR = np.power(cR*Seq,nR)
    Req = aR + bR/(1+affR)
    affO = np.power(cO*(Req+Seq),nO)
    Oeq =  aO + bO/(1+affO)
    # affOh = np.power(pars['cO']*Req,pars['nO'])
    # Oheq = pars['aO'] + pars['bO']/(1+affOh)

    # Calculate the second derivatives
    second_derivatives = []
    for symbol1 in symbolnames:
        row = []
        for symbol2 in symbolnames:
            row.append(Oeq.diff(myVars[symbol1],myVars[symbol2]))
        second_derivatives.append(row)

    # Create the second derivative matrix
    matrix = sym.Matrix(second_derivatives)

    return matrix

def second_derivative_matrix_loglog():

    # Output Fluorescence matrix
    affS = np.power(cS*I,nS)
    Seq = (aS + bS*affS)/(1+affS)
    affR = np.power(cR*Seq,nR)
    Req = aR + bR/(1+affR)
    affO = np.power(cO*(Req+Seq),nO)
    logOeq =  sym.log(aO + bO/(1+affO))
    # affOh = np.power(pars['cO']*Req,pars['nO'])
    # Oheq = pars['aO'] + pars['bO']/(1+affOh)

    # Calculate the second derivatives
    second_derivatives = []
    for symbol1 in symbolnames:
        row = []
        for symbol2 in symbolnames:
            row.append(myVars[symbol1]*myVars[symbol2]*logOeq.diff(myVars[symbol1],myVars[symbol2]))
        second_derivatives.append(row)

    # Create the second derivative matrix
    matrix = sym.Matrix(second_derivatives)
    return matrix   

def fourth_derivative_matrix_loglog_var():

    # Output Fluorescence matrix
    affS = np.power(cS*I,nS)
    Seq = (aS + bS*affS)/(1+affS)
    affR = np.power(cR*Seq,nR)
    Req = aR + bR/(1+affR)
    affO = np.power(cO*(Req+Seq),nO)
    logOeq =  sym.log(aO + bO/(1+affO))
    # affOh = np.power(pars['cO']*Req,pars['nO'])
    # Oheq = pars['aO'] + pars['bO']/(1+affOh)

    # Calculate the second derivatives
    second_derivatives = []
    for symbol1 in symbolnames:
        row = []
        for symbol2 in symbolnames:
            row.append(myVars[symbol1]**2*myVars[symbol2]**2*logOeq.diff(myVars[symbol1],myVars[symbol2],myVars[symbol1],myVars[symbol2]))
        second_derivatives.append(row)

    # Create the second derivative matrix
    matrix = sym.Matrix(second_derivatives)
    return matrix   


param_dict = {'aR':1.3,'aS':2,'aO':0.3,'cS':13,'cR':11,'cO':14,'bR':0.1,'bS':0.221,'bO':0.81,'nR':2.1,'nS':2.5,'nO':1.1,'I':7}

def numeric_matrix(matrix,pardict):
    M = sym.Matrix(matrix)
    for par in pardict:
        M = M.subs(myVars[par],param_dict[par])
    return M

def get_index(parname):
    return symbolnames.index(parname)


