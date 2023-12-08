## Initializing the Script
# Import necessary libraries
import os
import math as m
import pandas as pd
import numpy as np
import scipy.special as sp
from datetime import datetime


###############################################################################
##### MAIN FUNCTIONS USED IN THE CODE
# Define common function: error propagation equation
def err_prop_div(A,a,B,b):
    # Propagates errors associated with A (a) and B (b) to quotient = A/B
    return (A/B)*m.sqrt(((a/A)**2)+((b/B)**2))

## Data Transformation, Signal Calibration, and Model Equilibrium Condition
# Initial transformation will take the imported columns of data (arranged into
# distance and element intensities) and calculate ratio values (i.e. an element
# normalized by another, e.g. Sr/Si). The ratio values are corrected for 
# isotopic abundance and for each value an uncertainty estimate is calculated.
# Uncertainties for each isotope are listed as the last value of each column
# on each sheet. They are image-dependent, so they are likely different for 
# each image. These values are then transformed into concentration information 
# using an internal calibration. The transformed data form the base for 
# modelling the equilibrium Sr distribution for each crystal. The returned
# array holds the concentration profiles, their uncertainties, the model
# equilibrium data, the ratio of the observed/equilibrium value for each point
# along the profile, and the uncertainty associated with this value.
def sr_transformation(data,t):
    # Define some constants
    w = -26700.0 # Sr diffusion constant, units = mol/J
    R = 8.3145 # Universal gas constant, units = J/mol K
    # The last value of the distance vector is a NaN, so drop it
    distance = data["Distance"]
    lennie = len(distance) # Simple length argument
    sr_conc = data["Sr (ug/g)"]
    sr_unc = 0.0038*sr_conc
    an_num = data["An#"]
    an_unc = 0.0033*an_num
    sr_av = round(sum(sr_conc)/len(sr_conc)) # Calculate an avg. Sr conc
    # Generate intra-crystal partition coefficients, initialize
    d = []
    d_unc = []
    # Calculation utilizes constants, anorthite, and sample temp. data
    for x in range(lennie-1):
        d.append(m.exp((w*(an_num[x]-an_num[x+1]))/(R*t)))
        d_unc.append((-w/(R*t))*(m.exp((w*(an_num[x]-an_num[x+1]))/(R*t)))*\
                     m.sqrt((an_unc[x]**2)+(an_unc[x+1]**2)))
    d.append(0)
    d_unc.append(0)
    # Initialize the equilibrium Sr vector, to be calculated iteratively
    sr_equil = list(range(lennie))
    q = 1 # Set a counter for the iterative method
    sr_equil_av = sum(sr_equil)/len(sr_equil) # Used for convergence
    # The iterative method
    while sr_equil_av < sr_av: # Set convergence criterion
        sr_equil[0] = q+1
        for x in  range(1,lennie):
            sr_equil[x] = sr_equil[x-1]/d[x-1]
        sr_equil_av = round(sum(sr_equil)/len(sr_equil))
        q = q+0.5

    # Calculate uncertainty values for these model equilibrium conditions
    # At the same time, calculate the current (initial) Sr ratio value
    # and its uncertainty
    sr_equil_unc = []
    sr_ratio = []
    sr_ratio_unc = []
    # Populate these lists
    for x in range(lennie):
        sr_equil_unc.append(abs(sr_equil[x]*(w/(R*t))*an_unc[x]))
        sr_ratio.append(sr_conc[x]/sr_equil[x])
        sr_ratio_unc.append(err_prop_div(sr_conc[x],sr_unc[x],sr_equil[x],\
                                          sr_equil_unc[x]))
    colheads = ["Distance (um)","An#","An# 1-Sigma","D Vals","D 1-Sigma",\
                "Sr (ug/g)","Sr 1-Sigma","Eq. Sr (ug/g)","Eq. Sr 1-Sigma",\
                "Sr Ratio (Obs./Eq.)","Ratio 1-Sigma"]
    data_array = np.transpose(np.array([distance,an_num,\
                an_unc,d,d_unc,sr_conc,sr_unc,sr_equil,sr_equil_unc,sr_ratio,\
                sr_ratio_unc]))
    data_profile = pd.DataFrame(data_array, columns = colheads)
    return data_profile


## Regression Algorithm - Optional
# The ratio data generated in the last section constitutes the core dataset
# for further analysis. Next, a regression scheme is implemented to calculate
# the slope and associated uncertainties for individual points along the ratio
# profile. This is achieved using an optimized method that weighs probability 
# as well as the number of observations integrated for each point and uses 
# uncertainty in the slope calculation as a pseudo tie-breaker
# This is an optional part of the algorithm. Fourier Transform analysis 
# performed later in the script uses an instantaneous version of slope
# calculation without uncertainty estimation.
# Define functions to be used in this section
def regression_alg(input_data):
    # Define the functions needed to run the algorithm
    def comb_cons(l,mini):
        # For the creation of every possible combination of adjacent x values
        # l = length of the vector in question
        # mini = minimum number of adjacent cells (usually 2)
        for i in range(mini,l+1):
            a = range(0,l-i+1)
            b = range(0,i)
            c = np.add.outer(a,b)
            v1 = np.array([c[:,0],c[:,-1]])
            if i == mini:
                v = v1
            else:
                v = np.concatenate((v,v1),axis=1)
        return v.transpose()
    
    
    # For the following functions:
    # n = a number of indecies to test, as a list
    # reg_in = regression parameters arranged as a numpy array
    def sval(n,reg_in):
        x = []
        for y in range(len(n)):
            x.append(1/(reg_in[2][n[y]]**2))
        return sum(x)


    def sxval(n,reg_in):
        x = []
        for y in range(len(n)):
            x.append(reg_in[0][n[y]]/(reg_in[2][n[y]]**2))
        return sum(x)


    def syval(n,reg_in):
        x = []
        for y in range(len(n)):
            x.append(reg_in[1][n[y]]/(reg_in[2][n[y]]**2))
        return sum(x)


    def sxxval(n,reg_in):
        x = []
        for y in range(len(n)):
            x.append((reg_in[0][n[y]]**2)/(reg_in[2][n[y]]**2))
        return sum(x)


    def sxyval(n,reg_in):
        x = []
        for y in range(len(n)):
            x.append((reg_in[0][n[y]]*reg_in[1][n[y]])/(reg_in[2][n[y]]**2))
        return sum(x)


    def chisq(n,reg_in):
        s = sval(n,reg_in)
        sx = sxval(n,reg_in)
        sy = syval(n,reg_in)
        sxx = sxxval(n,reg_in)
        sxy = sxyval(n,reg_in)
        delta = (s*sxx)-(sx**2)
        slope = ((s*sxy)-(sx*sy))/delta
        inter = ((sxx*sy)-(sx*sxy))/delta
        chi = []
        for y in range(len(n)):
            chi.append(((reg_in[1][n[y]]-inter-(slope*reg_in[0][n[y]]))\
                        /reg_in[2][n[y]])**2)
        return sum(chi)


    def regres(n,reg_in):
        # Calculate slope and the uncertainty on that slope for a linear fit
        s = sval(n,reg_in)
        sx = sxval(n,reg_in)
        sy = syval(n,reg_in)
        sxx = sxxval(n,reg_in)
        sxy = sxyval(n,reg_in)
        delta = (s*sxx)-(sx**2)
        slope = ((s*sxy)-(sx*sy))/delta
        uncert = 2*m.sqrt(s/delta)
        return [slope,uncert]
    
    
    # The last 2 functions to calculate regression parameters and probabilities
    # of fit for each set of adjacent x values along the distance vector
    def regres_prob(reg_in):
        cols = reg_in.shape[1]
        a = comb_cons(cols,2)
        num = np.shape(a)[0]
        # Initialize the array of regression parameters
        A = np.zeros((a.shape[0],3))
        # Populate it
        for y in list(range(num)):
            v = list(a[y])
            u = list(range(v[0],v[1]+1))
            r = regres(u,reg_in)
            A[y][0] = r[0]
            A[y][1] = r[1]
            # Provide a probability of the fit
            if len(u) == 2:
                A[y][2] = 1
            else:
                A[y][2] = sp.gammainc(chisq(u,reg_in),(len(u)-2)/2)
        return np.append(A,a,axis=1)
    
    
    # The preceding function is used to calculate slopes, uncertainties, and 
    # fit probabilities for every combination of adjacent cells. The next step
    # of the algorithm filters out bad results and gives the best-fit solutions
    def regres_filter(regressions,dist,p):
        ns = regressions.shape[0]
        dcdx = np.zeros((dist,5))
        for x in list(range(dist)):
            vals = []
            fvals = []
            # Extract all applicable regression data from the  array with 
            # probabilities higher than the filter value
            for y in list(range(ns)):
                if regressions[y,3] <= x and regressions[y,4] >= x and\
                                                        regressions[y,2] > p:
                    vals.append(list(regressions[y,:]))
            vals = np.stack(vals,axis=0)
            # Calculate the maximum number of x elements used
            z = vals[:,4]-vals[:,3]
            maximum = np.amax(z)
            # Filter the results, taking only data generated using the maximum
            # number of input arguments
            for n in list(range(len(z))):
                if z[n] == maximum:
                    fvals.append(vals[n,:])
            # Saving these data: if multiple maxima are observed, takes the fit 
            # with the lowest uncertainty value
            if len(fvals) == 1:
                dcdx[x] = fvals[0]
            else:
                minimum = np.amin(fvals[1])
                result = np.where(fvals == minimum)
                coord = result[0][0]
                dcdx[x] = fvals[coord]
        return dcdx
    
    
    # The preceding function is used to process each sheet. For each value
    # along the distance of the profile, an optimized slope and uncertainty
    # estimate is extracted based upon the probability of the fit and the
    # number of observations integrated for that point
    data = input_data
    reg_in = np.transpose(np.stack((data["Distance"],data\
            ["Sr Ratio (Obs./Eq.)"],data["Ratio 1-Sigma"]),axis=1))
    lennie = reg_in.shape[1]
    p = 0.9 # Set a probability filter, very important parameter
    dcdx = regres_filter(regres_prob(reg_in),lennie,p)
    colheads = ["Slope", "Slope 1-Sigma", "Probability", "X1", "X2"]
    data_regressed = pd.DataFrame(dcdx, columns = colheads)
    return data_regressed
        

## The Diffusion Algorithm
# Define common function: calculation of diffusion coefficient
def calc_diff(an,t):
    # Utilizes the Giletti & Casserly (1994) formulation for calculating the 
    # diffusion coefficient of strontium in plagioclase
    R = 8.3145 # Universal gas constant, units = J/mol K
    t = t + 273.15 # Converts input temp. (Celsius) into Kelvin
    do = 10**(-8.18+(0.041*(1-an))) # Pre-exponential factor, m^2/s
    q = 277000 # Activation energy in J/mol
    D = do*m.exp(-q/(R*t)) # Units will be m^2/s
    return D


# Define common function: construct a logarithmic time vector
def build_timevec(initial,final,steps,threshold):
    log_t_st = m.log10(initial) # Initial time step from 0 - set to 1 hour
    log_t_fin = m.log10(final) # The time to total equilibrium
    time_in = np.logspace(log_t_st,log_t_fin,steps)
    time_diff = [time_in[i]-time_in[i-1] for i in range(1,len(time_in))]
    time_diff.insert(0,initial) # Make same as definitiion of log_t_st as float
    time_vec = [] # Initialize 
    crit_ind = [] 
    for x in range(len(time_diff)):
        if time_diff[x] < threshold:
            time_vec.append(time_diff[x])
        elif time_diff[x] >= threshold:
            time_vec.append(threshold)
            crit_ind.append(sum(time_diff[0:x+1]))
    while sum(time_vec) < final:
        time_vec.insert(-1,threshold)
    output = {"Time Vec": time_vec, "Critical Ind.": crit_ind}
    return output


# The diffusion algorithm, proper
def diffuse_sr(cal_data,t,input_data):
    # For the calculation of 1-dimensional two-way diffusion along a single 
    # profile to determine time to total slope equilibration
    # Import necessary data
    distance = np.array(input_data["Distance (um)"])
    dist_nominal = [str(i) for i in distance]
    dist_nominal.append("Time")
    dist_step = distance[1]/1000000 # Gives distance step in meters
    lennie = len(distance) # Usual length parameter
    sr_data = np.stack((input_data["Sr (ug/g)"],input_data["Sr 1-Sigma"]),\
                       axis=1)
    sr_equil_data = np.stack((input_data["Eq. Sr (ug/g)"],\
                              input_data["Eq. Sr 1-Sigma"]),axis=1)
    an_num = input_data["An#"]
    # Create the flux factors for diffusion modelling
    ks = np.zeros((lennie,4))
    for x in range(lennie):
        if x == 0: # Set first boundary condition
            ks[x,0] = 0
            ks[x,1] = 0
            ks[x,2] = (sr_equil_data[x,0]/(2*sr_equil_data[x+1,0]))+0.5
            ks[x,3] = (sr_equil_data[x+1,0]/(2*sr_equil_data[x,0]))+0.5
        elif x == lennie-1: # Set second boundary conditon
            ks[x,0] = (sr_equil_data[x,0]/(2*sr_equil_data[x-1,0]))+0.5
            ks[x,1] = (sr_equil_data[x-1,0]/(2*sr_equil_data[x,0]))+0.5
            ks[x,2] = 0
            ks[x,3] = 0
        else:
            ks[x,0] = (sr_equil_data[x,0]/(2*sr_equil_data[x-1,0]))+0.5
            ks[x,1] = (sr_equil_data[x-1,0]/(2*sr_equil_data[x,0]))+0.5
            ks[x,2] = (sr_equil_data[x,0]/(2*sr_equil_data[x+1,0]))+0.5
            ks[x,3] = (sr_equil_data[x+1,0]/(2*sr_equil_data[x,0]))+0.5
    # Calculate diffusion coefficients
    alphas = [] # Initialize the list
    for x in range(lennie):
        alphas.append(calc_diff(an_num[x],t))
    # The flux factors and diffusion coefficients are used to calculate the
    # maximum time step for modelling. First, find the maxima from each
    max_k = np.amax(ks)
    max_a = max(alphas)
    time_limit = (dist_step**2)/(2*max_k*max_a) # Output is in seconds
    # For diffusion, we use a time step equal to 95% of this limit
    time_step = round(0.95*time_limit)
    # Diffusion runs twice in this algorithm, the first time in a linear
    # manner (i.e. steps are of equal length), and the second time in a 
    # logarithmic manner. The goal of the first iteration is to find the 
    # total amount of time needed to diffuse the crystal to equilibrium, 
    # while the second is to diffuse it on a set logarithmic time path
    # First iteration
    # Set up a vector of boolean values, which record when a given part of
    # the profile reaches equilibrium
    is_equil = [False for i in range(lennie)]
    xx = 0 # Initialize a counter for the diffusion algorithm
    # Initialize arrays for saving diffused Sr profiles
    diffused = np.zeros((1,lennie)) # Recorder of diffused profiles
    diffy = np.zeros((1,lennie)) # To be overwritten each time step
    # Run diffusion
    print("Start diffusion 1")
    while not all(is_equil):
        xx = xx+1
        for x in range(lennie):
            if xx == 1:
                # Boundary conditions first
                if x == 0:
                    c = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *sr_data[x+1,0]
                    d = ((min(alphas[x:x+2])*time_step)/(dist_step**2))*\
                        sr_data[x,0]
                    diffused[0,x] = sr_data[x,0]+(ks[x,2]*c)-(ks[x,3]*d)
                elif x == lennie-1:
                    a = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *sr_data[x-1,0]
                    b = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))*\
                        sr_data[x,0]
                    diffused[0,x] = sr_data[x,0]+(ks[x,0]*a)-(ks[x,1]*b)
                else: # Then everything in between
                    a = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *sr_data[x-1,0]
                    b = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))*\
                        sr_data[x,0]
                    c = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *sr_data[x+1,0]
                    d = ((min(alphas[x:x+2])*time_step)/(dist_step**2))*\
                        sr_data[x,0]
                    diffused[0,x] = sr_data[x,0]+(ks[x,0]*a)-(ks[x,1]*b)+\
                        (ks[x,2]*c)-(ks[x,3]*d)
            else:
                if x == 0:
                    c = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x+1]
                    d = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x]
                    diffy[0,x] = diffused[xx-2,x]+(ks[x,2]*c)-(ks[x,3]*d)
                elif x == lennie-1:
                    a = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x-1]
                    b = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x]
                    diffy[0,lennie-1] = diffused[xx-2,x]+(ks[x,0]*a)-\
                        (ks[x,1]*b)
                else:
                    a = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x-1]
                    b = ((min(alphas[x-1:x+1])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x]
                    c = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x+1]
                    d = ((min(alphas[x:x+2])*time_step)/(dist_step**2))\
                        *diffused[xx-2,x]
                    diffy[0,x] = diffused[xx-2,x]+(ks[x,0]*a)-(ks[x,1]*b)+\
                        (ks[x,2]*c)-(ks[x,3]*d)
        if xx != 1:
            diffused = np.vstack([diffused,diffy])
        # Calculate ratio values for each intermediate output. The
        # algorithm will stop once the entire profile is within uncertainty
        # of one
        rsr = np.zeros((1,lennie))
        rsr_unc = np.zeros((1,lennie))
        for x in range(lennie):
            rsr[0,x] = diffused[xx-1,x]/sr_equil_data[x,0]
            rsr_unc[0,x] = err_prop_div(diffused[xx-1,x],sr_data[x,1],\
                        sr_equil_data[x,0],sr_equil_data[x,1])
        for x in list(range(len(is_equil))):
            if rsr[0,x]+(rsr_unc[0,x]) >1 and rsr[0,x]-(rsr_unc[0,x]) < 1:
                is_equil[x] = True
            else:
                is_equil[x] = False
        if int(xx/1000) == xx/1000:
            print(" ".join(["Iteration number:", str(xx), \
                    "-- Total diffusion time (s):", str(xx*time_step)]))
            print(" ".join(["Equilibrium:", str(rsr.sum()/lennie)]))
    print("End diffusion 1")
    num_steps = diffused.shape[0] # Get the number of steps performed
    # Generate a vector of times (summed), initialize
    time_vector_1 = np.zeros((1,num_steps))
    # Populate it
    for x in range(num_steps):
        time_vector_1[0,x] = (x+1)*time_step
    time_equil = time_vector_1[0,-1] # The last value is the total time
    # Combine the diffused data and the time_vector here to form a dataset
    # that will be used in the next step
    diffused_linear = np.hstack([diffused,time_vector_1.transpose()])
    # Second iteration
    # Build the time vector to be used
    time_hold = build_timevec(3600.0,time_equil,300,time_step)
    time_vector_2 = time_hold["Time Vec"]
    crit_indecies = time_hold["Critical Ind."]
    # Crit_indecies keeps track of the times where the algorithm should 
    # sample the diffused data along a logarithmic time scale
    # Re-initialize variables
    diffused_2 = np.zeros((1,lennie))
    diffy = np.zeros((1,lennie))
    save_data = np.zeros((1,lennie+1)) # For saving data
    savvy = np.zeros((1,lennie+1)) # To be overwritten
    xx = 0 # Counter for saving
    o = 0 # Iterable for time step
    yy = 0 # Summed time
    print("Start diffusion 2")
    while yy < crit_indecies[0]:
        xx = xx+1
        yy = sum(time_vector_2[0:o+1])
        for x in range(lennie):
            if xx == 1:
                # Boundary conditions first
                if x == 0:
                    c = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x+1,0]
                    d = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x,0]
                    diffused_2[0,x] = sr_data[x,0]+(ks[x,2]*c)-\
                        (ks[x,3]*d)
                elif x == lennie-1:
                    a = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x-1,0]
                    b = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x,0]
                    diffused_2[0,x] = sr_data[x,0]+(ks[x,0]*a)-\
                        (ks[x,1]*b)
                else: # Then everything in between
                    a = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x-1,0]
                    b = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x,0]
                    c = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x+1,0]
                    d = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*sr_data[x,0]
                    diffused_2[0,x] = sr_data[x,0]+(ks[x,0]*a)-\
                        (ks[x,1]*b)+(ks[x,2]*c)-(ks[x,3]*d)
            else:
                if x == 0:
                    c = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x+1]
                    d = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x]
                    diffy[0,x] = diffused_2[xx-2,x]+(ks[x,2]*c)-\
                        (ks[x,3]*d)
                elif x == lennie-1:
                    a = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x-1]
                    b = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x]
                    diffy[0,lennie-1] = diffused_2[xx-2,x]+\
                        (ks[x,0]*a)-(ks[x,1]*b)
                else:
                    a = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x-1]
                    b = ((min(alphas[x-1:x+1])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x]
                    c = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x+1]
                    d = ((min(alphas[x:x+2])*time_vector_2[o])/\
                         (dist_step**2))*diffused_2[xx-2,x]
                    diffy[0,x] = diffused_2[xx-2,x]+(ks[x,0]*a)-\
                        (ks[x,1]*b)+(ks[x,2]*c)-(ks[x,3]*d)
        o = o+1
        if xx == 1:
            save_data = diffused_2[xx-1,:]
            save_data = np.append(save_data,yy)
        elif xx != 1:
            diffused_2 = np.vstack([diffused_2,diffy])
            savvy = diffused_2[xx-1,:]
            savvy = np.append(savvy,yy)
            save_data = np.vstack([save_data,savvy])
    print("End diffusion 2")
    # The preceding diffused over a log scale up until the first critical
    # index. The algorithm will now assess the original diffused matrix for
    # data after the critical index to append to the newly created matrix
    time_last = save_data[-2,-1] # Total amount of time diffused in log scale
    times_over = [] # Initialize a list to save indecies
    for x in range(num_steps):
        if time_last < diffused_linear[x,-1]:
            times_over.append(x)
    time_link = times_over[0] # This is the index we need to use
    diffused_to_append = diffused_linear[time_link:,:]
    data_diffused = np.vstack([save_data,diffused_to_append])
    data_diffused = pd.DataFrame(data_diffused,columns=dist_nominal)
    return data_diffused


def ratio_regression(input_1,input_2):
    dist_vector = input_1["Distance (um)"]
    dist_nominal = [str(i) for i in dist_vector]
    lennie = len(dist_vector)
    sr_equil = input_1["Eq. Sr (ug/g)"]
    diffused_sr = np.array(input_2[dist_nominal])
    num_runs = diffused_sr.shape[0]
    ratio_matrix = np.zeros((num_runs,lennie))
    slopes = np.zeros((num_runs,lennie))
    for x in range(num_runs):
        for y in range(lennie):
            ratio_matrix[x,y] = diffused_sr[x,y]/sr_equil[y]
    for x in range(num_runs):
        for y in range(lennie):
            if y == 0:
                slopes[x,y] = (ratio_matrix[x,y+1]-ratio_matrix[x,y])/\
                    (dist_vector[y+1]-dist_vector[y])
            elif y == lennie-1:
                slopes[x,y] = (ratio_matrix[x,y]-ratio_matrix[x,y-1])/\
                    (dist_vector[y]-dist_vector[y-1])
            else:
                slopes[x,y] = (ratio_matrix[x,y+1]-ratio_matrix[x,y-1])/\
                    (dist_vector[y+1]-dist_vector[y-1])
    return {"Ratio Vals": ratio_matrix, "Slope Vals": slopes}


def data_saver(sample_name,input_data_1,input_data_2,t):
    with pd.ExcelWriter(f"{sample_name}_diffused.xlsx") as writer:
        input_data_1.to_excel(writer,sheet_name="Transformed SCAPS Data")
        an_num = input_data_1["An#"]
        alpha_vals = []
        lennie = len(an_num)
        for x in range(lennie):
            alpha_vals.append(calc_diff(an_num[x],t))
        input_data_2.to_excel(writer,sheet_name="Diffusion Data - Conc")
        dist_vector = input_data_1["Distance (um)"]
        dist_nominal = [str(i) for i in dist_vector]
        alphas = pd.Series(data=alpha_vals,index=dist_nominal)
        alphas.to_excel(writer,sheet_name="Diffusion Coefficients")
        colheads = dist_nominal.append("Time")
        ratios1 = ratio_regression(input_data_1,input_data_2)["Ratio Vals"]
        ratios2 = ratio_regression(input_data_1,input_data_2)["Slope Vals"]
        pd.DataFrame(ratios1,columns=colheads).to_excel(writer,sheet_name=\
                                            "Diffusion Data - Ratios")
        pd.DataFrame(ratios2,columns=colheads).to_excel(writer,sheet_name=\
                                            "Diffusion Data - Slopes")
        

###############################################################################
##### MAIN CODE FOR RUNNING SAMPLES
## Retrieve the Data
# Set the directory to the file path
os.chdir(r"C:\Users\dcoultha\Documents\Data\Sr-in-Plag_2021\Python\Input Files")
# Define dataset and associated names
input_file = "Ruapehu.xlsx" # Define file to work with
# sample = input_file.removesuffix(".xlsx") # Recover sample set name
project = pd.ExcelFile(input_file) # Reads in the file
sheets = project.sheet_names # Index of individual analyses stored in the file
sample_data = sheets.pop(-1) # Removes the calibration and temperature data
# Start a count to see how long it takes the code to run
startTime = datetime.now()
cal_data = pd.read_excel(project,sample_data)
temps = cal_data["T Vals"]
os.chdir(r"C:\Users\dcoultha\Documents\Data\Sr-in-Plag_2021\Python\Diffusion_Output\Ruapehu")
for i in sheets:
    print(i)
    profile = sr_transformation(pd.read_excel(project,i),temps[sheets.\
                                                    index(i)])
    print(datetime.now() - startTime)
    diff_data = diffuse_sr(cal_data,temps[sheets.index(i)],profile)
    print(datetime.now() - startTime)
    data_saver(i,profile,diff_data,temps[sheets.index(i)])
    print(datetime.now() - startTime)