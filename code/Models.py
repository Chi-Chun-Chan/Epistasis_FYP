import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy.optimize import root_scalar


# plots make sense
#should investigate the relationships between inducer and sensor (calculated using regression of inducer )
#function to do subsequent non linear regression is :
#%%
#we will now create model functions for each of the steady state values of Sensor, Regulator and Output.
#Henceforth referred to as S, R and O

#hill function model
def init_model(I_conc,A_s,B_s,C_s,N_s, A_r,C_r,N_r,A_o,C_o,N_o):
    Sensor= (A_s+B_s*(C_s*I_conc)**N_s)/(1+np.power(C_s*I_conc,N_s))
    Regulator = (A_r)/(1+ np.power(C_r*Sensor,N_r))
    Sens_Reg = Sensor + Regulator
    Output = (A_o)/(1+np.power(C_o*Sensor,N_o))
    Stripe = (A_o)/(1+np.power(C_o*Sens_Reg,N_o))
    return Sensor, Regulator, Output, Stripe

def model_leaky(I_conc,A_s,B_s,C_s,N_s,L_r,A_r,C_r,N_r,L_o,A_o,C_o,N_o):
    Sensor= (A_s+B_s*(C_s*I_conc)**N_s)/(1+np.power(C_s*I_conc,N_s))
    Regulator = L_r+(A_r)/(1+ np.power(C_r*Sensor,N_r))
    Sens_Reg = Sensor + Regulator
    Output = (A_o)/(1+np.power(C_o*Sensor,N_o))
    Stripe = L_o+(A_o)/(1+np.power(C_o*Sens_Reg,N_o))
    return Sensor, Regulator, Output, Stripe

#example input to model_hill
#assumes degredation of lacI, TetR and GFP are constant
#params_dict={"sen_params":{"As":1,"Bs":1,"Cs":1,"Ns":1},"reg_params":{"Ar":1,"Br":1,"Cr":1,"Nr":1},"out_h_params":{"Ah":1,"Bh":1,"Ch":1},"out_params":{"Fo":1,"Ao":1,"Bo":1,"Co":1,"No":1}}
#%%
class model_hill:
    example_dict={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1,"F_o":1},"free_params":{}}
    def __init__(self,params_list:list,I_conc):
        self.params_list=params_list
        self.I_conc=I_conc
        #self.example_dict_model_1={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{"A_h":1,"B_h":1,"C_h":1},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}
        self.example_dict={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}
        
        self.n_parameters_1=16
        self.n_parameters_2=13
    def get_dict(self):
        return self.example_dict_model_2
    @staticmethod    
    def model_old(params_list,I_conc):
        correct_length=16
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        # As=params_dict['sen_params']['As']
        # Bs=params_dict['sen_params']['Bs']
        # Cs=params_dict['sen_params']['Cs']
        # Ns=params_dict['sen_params']['Ns']
        # Ar=params_dict['reg_params']['Ar']
        # Br=params_dict['reg_params']['Br']
        # Cr=params_dict['reg_params']['Cr']
        # Nr=params_dict['reg_params']['Nr']
        # Ah=params_dict['out_h_params']['Ah']
        # Bh=params_dict['out_h_params']['Bh']
        # Ch=params_dict['out_h_params']['Ch']
        # Fo=params_dict['out_params']['Fo']
        # Ao=params_dict['out_params']['Ao']
        # Bo=params_dict['out_params']['Bo']
        # Co=params_dict['out_params']['Co']
        # No=params_dict['out_params']['No']
        
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return 0
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
        #out_half
        A_h=params_list[8]
        B_h=params_list[9]
        C_h=params_list[10]
        #output
        A_o=params_list[11]
        B_o=params_list[12]
        C_o=params_list[13]
        N_o=params_list[14]
        #free
        F_o=params_list[15]
        
        Sensor = A_s+B_s*np.power(C_s*I_conc,N_s)
        Sensor /= 1+np.power(C_s*I_conc,N_s)

        Regulator = B_r/(1+np.power(C_r*Sensor,N_r))
        Regulator += A_r

        Output_half = B_h/(1+np.power(C_h*Sensor,N_o))
        Output_half += A_h

        Output = A_o*A_h + B_o*B_h/(1+np.power(C_h*(Sensor+C_o*Regulator),N_o))
        Output*=F_o
        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output
    @staticmethod    
    def model_new_WT(params_list,I_conc): #reformulated so that output half and output are same parameters with F_o to scale
        correct_length=14 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
   
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0] #assume that all parameters are un-logged prior
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
        #out_half
        
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        C_k=params_list[11]
        N_o=params_list[12]
        
        #free
        F_o=params_list[13]
        
        Sensor = A_s+B_s*np.power(C_s*I_conc,N_s)
        Sensor /= 1+np.power(C_s*I_conc,N_s)

        Regulator = B_r/(1+np.power(C_r*Sensor,N_r))
        Regulator += A_r

        Output_half = B_o/(1+np.power(C_o*Sensor,N_o))
        Output_half += A_o

        # Output = A_o + B_o/(1+np.power(((C_o*Regulator)+(C_k*Sensor)),N_o))
        # Output*=F_o

        Output = A_o + B_o/(1+np.power(C_o*((C_k*Sensor) + Regulator),N_o))
        Output*=F_o
        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output
    @staticmethod    
    def model(params_list,I_conc): #reformulated so that output half and output are same parameters with F_o to scale
        correct_length=13 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
   
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
        #out_half
        
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        N_o=params_list[11]
        
        #free
        F_o=params_list[12]
        
        Sensor = A_s+B_s*np.power(C_s*I_conc,N_s)
        Sensor /= 1+np.power(C_s*I_conc,N_s)

        Regulator = B_r/(1+np.power(C_r*Sensor,N_r))
        Regulator += A_r

        Output_half = B_o/(1+np.power(C_o*Sensor,N_o))
        Output_half += A_o

        Output = A_o + B_o/(1+np.power(C_o*(Sensor+Regulator),N_o))
        Output*=F_o
        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output
    @staticmethod    
    def model_muts(params_list,I_conc): #reformulated for 
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        MA_s=params_list[4]
        MB_s=params_list[5]
        MC_s=params_list[6]
        MN_s=params_list[7]
        
        #regulator
        A_r=params_list[8]
        B_r=params_list[9]
        C_r=params_list[10]
        N_r=params_list[11]
        MA_r=params_list[12]
        MB_r=params_list[13]
        MC_r=params_list[14]
        MN_r=params_list[15]
    
        #out_half
        
        #output
        A_o=params_list[16]
        B_o=params_list[17]
        C_o=params_list[18]
        C_k=params_list[19]
        N_o=params_list[20]
        F_o=params_list[21]

        MA_o=params_list[22]
        MB_o=params_list[23]
        MC_o=params_list[24]
        MN_o=params_list[25]
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= (1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s)))

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power(((MC_o*C_o)*((C_k*Sensor) + Regulator)),(MN_o*N_o)))
        Output*= F_o

        return Sensor,Regulator,Output_half, Output
    
    @staticmethod    
    def model_muts2(params_list,I_conc): #reformulated for 
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
    
        #out_half
    
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        C_k=params_list[11]
        N_o=params_list[12]
        F_o=params_list[13]

        MA_s=params_list[14]
        MB_s=params_list[15]
        MC_s=params_list[16]
        MN_s=params_list[17]
        MA_r=params_list[18]
        MB_r=params_list[19]
        MC_r=params_list[20]
        MN_r=params_list[21]
        MA_o=params_list[22]
        MB_o=params_list[23]
        MC_o=params_list[24]
        MN_o=params_list[25]
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= (1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s)))

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power(((MC_o*C_o)*((C_k*Sensor) + Regulator)),(MN_o*N_o)))
        Output*= F_o

        return Sensor,Regulator,Output_half, Output

    def model_single_muts(params_list:list,I_conc,mutant): #reformulated for 
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        MA_s=params_list[4]
        MB_s=params_list[5]
        MC_s=params_list[6]
        MN_s=params_list[7]
        
        #regulator
        A_r=params_list[8]
        B_r=params_list[9]
        C_r=params_list[10]
        N_r=params_list[11]
        MA_r=params_list[12]
        MB_r=params_list[13]
        MC_r=params_list[14]
        MN_r=params_list[15]
    
        #out_half
        
        #output
        A_o=params_list[16]
        B_o=params_list[17]
        C_o=params_list[18]
        N_o=params_list[19]
        F_o=params_list[20]
        MA_o=params_list[21]
        MB_o=params_list[22]
        MC_o=params_list[23]
        MN_o=params_list[24]
        #free
        MF_o=params_list[25]

        if mutant == 'S':
            MA_o = 1
            MB_o = 1
            MC_o = 1
            MN_o = 1
            MF_o = 1
            MA_r = 1
            MB_r = 1
            MC_r = 1
            MN_r = 1
        elif mutant == 'R':
            MA_o = 1
            MB_o = 1
            MC_o = 1
            MN_o = 1
            MF_o = 1
            MA_s = 1
            MB_s = 1
            MC_s = 1
            MN_s = 1
        else:
            MA_s = 1
            MB_s = 1
            MC_s = 1
            MN_s = 1
            MA_r = 1
            MB_r = 1
            MC_r = 1
            MN_r = 1
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= 1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power((MC_o*C_o)*(Sensor+Regulator),(MN_o*N_o)))
        Output*= (MF_o*F_o)

        return Sensor,Regulator,Output_half, Output

    
    @staticmethod
    def modelWT(params_list,I_conc): #reformulated so that output half and output are same parameters with F_o to scale
        correct_length=13 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
   
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=10**params_list[0]
        B_s=10**params_list[1]
        C_s=10**params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=10**params_list[4]
        B_r=10**params_list[5]
        C_r=10**params_list[6]
        N_r=params_list[7]
        #out_half
        
        #output
        A_o=10**params_list[8]
        B_o=10**params_list[9]
        C_o=10**params_list[10]
        N_o=params_list[11]
        
        #free
        F_o=params_list[12]
        
        Sensor = A_s+B_s*np.power(C_s*I_conc,N_s)
        Sensor /= 1+np.power(C_s*I_conc,N_s)

        Regulator = B_r/(1+np.power(C_r*Sensor,N_r))
        Regulator += A_r

        Output_half = B_o/(1+np.power(C_o*Sensor,N_o))
        Output_half += A_o

        Output = A_o + B_o/(1+np.power(C_o*(Sensor+Regulator),N_o))
        Output*=F_o
        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output


#%%
    
class model_thermodynamic:
    example_dict={"sen_params":{"P_b":1,"P_u":1,"K_12":1,"C_pa":1,"A_s":1},"reg_params":{"P_r":1,"C_pt":1,"K_t":1,"A_r":1},"out_h_params":{},"out_params":{"P_o":1,"C_pl":1, "K_l":1,"A_o":1},"free_params":{},"fixed_params":{"F_o":1}}
    def __init__(self,params_list:list,I_conc):
        self.params_list=params_list
        self.I_conc=I_conc
        self.example_dict={"sen_params":{"P_b":1,"P_u":1,"K_12":1,"C_pa":1,"A_s":1},"reg_params":{"P_r":1,"C_pt":1,"K_t":1,"A_r":1},"out_h_params":{},"out_params":{"P_o":1,"C_pl":1, "K_l":1,"A_o":1},"free_params":{},"fixed_params":{"F_o":1}}
    # P_b = Kp_bent*[P]
    # K_t = summarises dimirasation and tetracycline binding 
    # P_p = K_p*[P]
    # A_n is alpha divided by beta
    #free params are those that vary for everty mutant 
    #fixed params are those that are constant for every node and are determined via WT
    #constraints
    #>C_pa>1
    #0<C_pt<1
    #0<C_pl<1
    
    # thermodynamics model
    @staticmethod
    def model(params_list:list,I_conc):
        correct_length=14
        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return 0

        # a_s, a_r, a_o represent the production rates divided by the degradation rates
        # I represents arabinose
        # P represents concentration of polymerase
        # Ki, Kii, Kps, Kpr, Ks, Kpo, K_lacI represent the binding affinity of the interaction
        # C_pi, C_ps, C_po_lacI represent the level of cooperative binding between activator or repressor with polymerase
        # sensor
        P_b=params_list[0]
        P_u=params_list[1]
        K_12=params_list[2]
        C_pa=params_list[3]
        A_s=params_list[4]
        # regulator
        P_r=params_list[5]
        C_pt=params_list[6]
        K_t=params_list[7]
        A_r=params_list[8]
        # output
        P_o=params_list[9]
        C_pl=params_list[10]
        K_l=params_list[11]
        A_o=params_list[12]
        # shared
        
        # fixed
        F_o=params_list[13]
        I=I_conc

        Sensor = P_b+C_pa*P_u*K_12*I**2
        Sensor /= 1+P_b+C_pa*P_u*K_12*I**2+K_12*I**2
        Sensor *= A_s

        Regulator = P_r+C_pt*P_r*K_t*Sensor**2
        Regulator /= 1+P_r+C_pt*P_r*K_t*Sensor**2+K_t*Sensor**2
        Regulator *= A_r

        Output_half = P_r+C_pl*K_l*P_r*Sensor**2
        Output_half /= 1+P_r+C_pl*K_l*P_r*Sensor**2+K_l*Sensor**2
        Output_half *= A_o
        
        Output = P_r+C_pl*K_l*P_r*(Sensor+Regulator)**2
        Output /= 1+P_r+C_pl*K_l*P_r*(Sensor+Regulator)**2+K_l*(Sensor+Regulator)**2
        Output *= A_o
        Output *= F_o

        return Sensor, Regulator, Output_half, Output
#%%
class model_hill_shaky:
    def __init__(self,params_list:list,I_conc):
        self.params_list=params_list
        self.I_conc=I_conc
        self.example_dict={"sen_params":{"A_s":500,"B_s":25000,"C_s":1200,"N_s":1},"reg_params":{"A_r":3000,"B_r":10000,"C_r":0.00001,"N_r":0.01},"out_h_params":{},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}
        #params_list = dict_to_list(example_dict)
        self.correct_length=16
    @staticmethod
    def model(params_list:list,I_conc):
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        #creates variables described in params_dict 
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        N_o=params_list[11]
        #free
        F_o=params_list[12]
        
        Sensor = np.array([])
        Regulator = np.array([])
        Output_half = np.array([])
        Output = np.array([])
        #initial conditions assumed as steady state with no inducer present
        S0 = A_s 
        R0 = B_r/(1+np.power(C_r*S0,N_r)) + A_r
        H0 = B_o/(1+np.power(C_o*S0,N_o)) + A_o
        O0 = (B_o/(1+np.power(C_o*(S0+R0),N_o)))*F_o
        #arbitrary time point to integrate ODE up to
        t = np.linspace(0,2,2) #0,1,2
        #define system of ODEs to be solved by odeint, for a each inducer concentration
        def ODE_S(S, t, conc):
            #S for sensor concentration at time t, prod for production
            S_prod = A_s+B_s*np.power(C_s*conc,N_s)
            S_prod /= 1+np.power(C_s*conc,N_s)
            #change in S concentration w.r.t. time, deg for degredation rate
            dSdt = S_prod - S
            return dSdt
        
        def ODE_R(R,t, S):
            R_prod = A_r*(1+B_r/(1+np.power(C_r*S,N_r)))
            dRdt = R_prod - R
            return dRdt
        
        def ODE_H(H,t, S):
            OH_prod = A_o + B_o/(1+np.power(C_o*(S),N_o))
            dOdt = OH_prod - H
            return dOdt
        
        def ODE_O(O,t, S_R):
            O_prod = A_o + B_o/(1+np.power(C_o*(S_R),N_o))
            dOdt = O_prod - (O)*F_o
            return dOdt

        for conc in I_conc:
            S = odeint(ODE_S, S0, t, args = (conc,))[-1]
            Sensor = np.append(Sensor, S)
            R = odeint(ODE_R, R0, t, args = (S,))[-1]
            Regulator = np.append(Regulator, R)
            H = odeint(ODE_H, H0, t, args = (S,))[-1]
            Output_half = np.append(Output_half, H)
            O = odeint(ODE_O, O0, t, args = (S+R,))[-1]
            Output = np.append(Output, O)
        return Sensor,Regulator , Output_half, Output
    #%%

#function to minimize while fitting, expects a dictionary of parameters corresponding to the model of interest, 
def min_fun(params_list:list,data,model_type):
        log_sen = np.log10(data.Sensor)
        log_reg = np.log10(data.Regulator)
        log_out = np.log10(data.Output)
        log_stripe = np.log10(data.Stripe)   
        ind = data.S
        Sensor_est, Regulator_est, Output_est, Stripe_est = model_type(params_list,I_conc=ind)
        log_sen_est = np.log10(Sensor_est)
        log_reg_est = np.log10(Regulator_est)
        log_out_est = np.log10(Output_est)
        log_stripe_est = np.log10(Stripe_est)

        #need to know what variables exist for each given mutant
        if "Mutant_ID" in data:
            mutant_id=data.Mutant_ID[0]
            if mutant_id.startswith("Sensor"):
                #need to ignore reg and output in fitting
                log_reg,log_reg_est,log_out,log_out_est=0,0,0,0
            # #if mutant_id.startswith("Regulator"):
            #     #need to ignore reg and output in fitting
            #     log_sen,log_sen_est,log_out,log_out_est=0,0,0,0
            # #if mutant_id.startswith("Output"):
            #     #need to ignore reg and sensor in fitting
            #     log_reg,log_reg_est,log_sen,log_sen_est=0,0,0,0
        
        result = np.power((log_sen - log_sen_est), 2)
        result += np.power((log_reg - log_reg_est), 2)
        result += np.power((log_out - log_out_est), 2)
        result += np.power((log_stripe - log_stripe_est), 2)
        return np.sum(result)

#only minimises to sensor and stripe
# def min_fun(params_list:list,data,model_type):
#         log_sen = np.log10(data.Sensor)
#         log_reg = np.log10(data.Regulator)
#         log_out = np.log10(data.Output)
#         log_stripe = np.log10(data.Stripe)   
#         ind = data.S
#         Sensor_est, Regulator_est, Output_est, Stripe_est = model_type(params_list,I_conc=ind)
#         log_sen_est = np.log10(Sensor_est)
#         log_reg_est = np.log10(Regulator_est)
#         log_out_est = np.log10(Output_est)
#         log_stripe_est = np.log10(Stripe_est)

#         #need to know what variables exist for each given mutant
#         # if "Mutant_ID" in data:
#         #     mutant_id=data.Mutant_ID[0]
#         #     if mutant_id.startswith("Sensor"):
#                 #need to ignore reg and output in fitting
#         log_reg,log_reg_est,log_out,log_out_est=0,0,0,0
#             # #if mutant_id.startswith("Regulator"):
#             #     #need to ignore reg and output in fitting
#             #     log_sen,log_sen_est,log_out,log_out_est=0,0,0,0
#             # #if mutant_id.startswith("Output"):
#             #     #need to ignore reg and sensor in fitting
#             #     log_reg,log_reg_est,log_sen,log_sen_est=0,0,0,0
        
#         result = np.power((log_sen - log_sen_est), 2)
#         result += np.power((log_reg - log_reg_est), 2)
#         result += np.power((log_out - log_out_est), 2)
#         result += np.power((log_stripe - log_stripe_est), 2)
#         return np.sum(result)










#model_hill_shakey: 
#uses hill functions as in model_hill, but does not assume that a steady state has been reached in the system
#example input to model_hill_shakey:

#%% redefine model_hill_shaky
#I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
#%%
# Competition Model

# F(S,R(S),O(S,R(S)))=0
# F should be a function that takes one variable (S), and it's 0 when that's true 

# f(I), in our case I, represents production of S
# g(S) represents protduction of R
# K, d - constants 

class CompDeg:
    def __init__(self,params_list:list,I_conc):
        self.params_list=params_list
        self.I_conc=I_conc
        self.example_dict_model={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"B_r":1,"C_r":1,"N_r":1},"out_h_params":{},"out_params":{"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1, "K":1}}
        self.n_parameters_2=12

    @staticmethod    
    def model(params_list,I_conc):             #reformulated so that output half and output are same parameters with F_o to scale
        correct_length=12
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        # As=params_dict['sen_params']['As']
        # Bs=params_dict['sen_params']['Bs']
        # Cs=params_dict['sen_params']['Cs']
        # Ns=params_dict['sen_params']['Ns']
        # Ar=params_dict['reg_params']['Ar']
        # Br=params_dict['reg_params']['Br']
        # Cr=params_dict['reg_params']['Cr']
        # Nr=params_dict['reg_params']['Nr']
        # Ah=params_dict['out_h_params']['Ah']
        # Bh=params_dict['out_h_params']['Bh']
        # Ch=params_dict['out_h_params']['Ch']
        # Fo=params_dict['out_params']['Fo']
        # Ao=params_dict['out_params']['Ao']
        # Bo=params_dict['out_params']['Bo']
        # Co=params_dict['out_params']['Co']
        # No=params_dict['out_params']['No']
        # K =params_dict['out_params']['K']
        # Deg =params_dict['out_params']['Deg']

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return 0
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        B_r=params_list[4]
        C_r=params_list[5]
        N_r=params_list[6]  
        #output
        B_o=params_list[7]
        C_o=params_list[8]
        N_o=params_list[9] 
        #free
        F_o=params_list[10]
        K= params_list[11] #affinity of S, R & O for protease
        #Deg = params_list[12] #average linear degredation term for S, R & O
        
        #since new system of equasions must be defined for each I_conc, a for loop is used to redefine the functions.
        Sensor = np.array([])
        Regulator = np.array([])
        Output_half = np.array([])
        Output = np.array([])
        for I in I_conc:
            #S PRODUCED is constant for a given I conc
            S_prod = A_s+B_s*np.power(C_s*I,N_s)
            S_prod /= 1+np.power(C_s*I,N_s)

            #Amount of R PRODUCED depends only on the amount of Sensor at a given time
            def R_prod(S):
                R_prod = B_r/(1+np.power(C_r*S,N_r))
                #R_prod += A_r
                return R_prod
            
            #Since R depends on S only,
            #The amount of O PRODUCED in the full or half network depends on the amount of S at a given time. 
            def H_prod(S):
                H_prod = B_o/(1+np.power(C_o*S,N_o))
                #H_prod += A_o
                return H_prod

            #Amount of R in the full network (as a function of S)
            def R_SRO(S):
                R__SRO = np.divide(np.multiply(R_prod(S),S),S_prod)
                return R__SRO
            
            #Amount of R in I --> S --| R should be the same, but a new function is defined for clarity later on
            def R_SR(S):
                return R_SRO(S)

            #therefore F_SR(S) is a function of S with S* in F_SR(S*)= 0 equal to S at steady state (of network with Output missing)
            def F_SR(S):
                return S_prod - S*K/(1+K*(S+R_SR(S)))-S
            
            #Amount of O in I --> S --| R is similar to R_SR, but with a different production term
            def O_SO(S):
                return (H_prod(S)*S)/S_prod
            
            #therefore F_SO(S) is a function of S with S* in F_SO(S*)= 0 equal to S at steady state (of network with Regulator missing)
            def F_SO(S):
                O = O_SO(S)
                return S_prod - S*K/(1+K*(S+O))-S

            #function calculating production of O in full network, for a given S
            def O_prod(S):
                R__SRO = R_SRO(S)
                O_prod = B_o/(1+np.power(C_o*(S+R__SRO),N_o))
                return O_prod
            
            #Amount of O at one time can be written as a quadratic function of S and R with 2 real roots, one positive and one negative
            def O_SRO(S):
                R = R_SRO(S)
                #write quadratic function for O in form aO^2+bO-c
                #a & c > 0 
                a = K
                b = 2+K*(S+R)-O_prod(S)*K
                c = -(1+K*(S+R))
                return max((-b+np.power((np.power(b,2)-4*a*c),0.5))/(2*a), (-b-np.power((np.power(b,2)-4*a*c),0.5))/(2*a))

            #Amount of S at steady state in full network can now be written as a function F_SRO(S) with a root equal to the predicted amount of S at this steady state
            def F_SRO(S):
                R = R_SRO(S)
                O = O_SRO(S)
                return S_prod - S/(1+K*(S+R+O)) - S
            
            Sensor = np.append(Sensor, S_prod/(2+K - K))

            S_SR = root_scalar(F_SR , method='secant', xtol=1e-3, x0 = 0, x1 = 2000).root
            Regulator = np.append(Regulator, R_SR(S_SR))

            S_SO = root_scalar(F_SO , x0 = 0, x1 = 2000, method='secant', xtol=1e-3).root
            Output_half = np.append(Output_half, (O_SO(S_SO)))

            S_SRO = root_scalar(F_SRO , x0 = 0, x1 = 2000, method='secant', xtol=1e-3).root
            Output =np.append(Output,F_o*(O_SRO(S_SRO)))

        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output

#TESTING model for right shape
from data_wrangling import meta_dict
from itertools import chain
import matplotlib.pyplot as plt

def dict_to_list(params_dict,return_keys=False):
    if return_keys==True:
       a=[list(i.keys()) for i in list(params_dict.values())]
    elif return_keys==False:
        a=[list(i.values()) for i in list(params_dict.values())]
        return list(chain.from_iterable(a))

#%%
def test():
    dict_model_thermo={"sen_params":{"P_b":1,"P_u":1,"K_12":1,"C_pa":1,"A_s":1},"reg_params":{"P_r":1,"C_pt":1,"K_t":1,"A_r":1},"out_h_params":{},"out_params":{"P_o":1,"C_pl":1, "K_l":1,"A_o":1},"free_params":{},"fixed_params":{"F_o":1}}

    params_list =  [9.06e-02,3e2, 3.48e+01, 9.00e-10, 1.050e+04, 1,3e-2, 7.711e-03, 6.750e+03 ,1, 6.006e-02, 3.133e+04 ,1.830e+00 ,3.513e+05] #dict_to_list(dict_model_thermo) #

    model_hill_shaky.model(params_list=params_list, I_conc=meta_dict['WT'].S)
    fig, ((axS,axR),(axH, axO)) = plt.subplots(2,2)
    I_concs = meta_dict['WT'].S
    S = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[0]
    R = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[1]
    H = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[2]
    O = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[3]
    axS.plot(I_concs,S)
    axR.plot(I_concs,R)
    axH.plot(I_concs,H)
    axO.plot(I_concs,O)
    for ax in fig.get_axes():
        ax.set_yscale('log')
        ax.set_xscale('log')
#%%

# #%%
# dict_model_thermo={"sen_params":{"P_b":1,"P_u":1,"K_12":1,"C_pa":1,"A_s":1},"reg_params":{"P_r":1,"C_pt":1,"K_t":1,"A_r":1},"out_h_params":{},"out_params":{"P_o":1,"C_pl":1, "K_l":1,"A_o":1},"free_params":{},"fixed_params":{"F_o":1}}

# params_list =  [9.06e-02,3e2, 3.48e+01, 9.00e-10, 1.050e+04, 1,3e-2, 7.711e-03, 6.750e+03 ,1, 6.006e-02, 3.133e+04 ,1.830e+00 ,3.513e+05] #dict_to_list(dict_model_thermo) #

# model_hill_shaky.model(params_list=params_list, I_conc=meta_dict['WT'].S)
# fig, ((axS,axR),(axH, axO)) = plt.subplots(2,2)
# I_concs = meta_dict['WT'].S
# S = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[0]
# R = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[1]
# H = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[2]
# O = model_thermodynamic.model(params_list=params_list, I_conc=meta_dict['WT'].S)[3]
# axS.plot(I_concs,S)
# axR.plot(I_concs,R)
# axH.plot(I_concs,H)
# axO.plot(I_concs,O)
# for ax in fig.get_axes():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
# #%%


