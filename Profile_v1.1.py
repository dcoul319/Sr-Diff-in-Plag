# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:30:49 2022

@author: mumob
"""

import os
import os.path
import pandas as pd
import numpy as np
import math
from scipy.signal import tukey
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import tkinter as tk
from tkinter import filedialog as fd
from scipy.optimize import curve_fit
from matplotlib.widgets import Button, RangeSlider, Slider, RadioButtons, CheckButtons, TextBox

class Profile:
    
    def __derivate_1D(self, vector):
        if vector.any():
            dim = len(vector)
            slopes = np.zeros(dim)
            for i in range(dim):
                if i == dim-1:
                    slopes[i] = (vector[i]-vector[i-1])/\
                        (self.distance[i]-self.distance[i-1])
                else:
                    slopes[i] = (vector[i+1]-vector[i])/\
                        (self.distance[i+1]-self.distance[i])
        else:
            slopes = vector
        return slopes
    
    def __derivate_2D(self, array):
        if array.any():
            dims = array.shape
            slopes = np.zeros(dims)
            for i in range(dims[0]):
                for j in range(dims[1]):
                    if j == dims[1]-1:
                        slopes[i][j] = (array[i][j]-array[i][j-1])/\
                            (self.distance[j]-self.distance[j-1])
                    else:
                        slopes[i][j] = (array[i][j+1]-array[i][j])/\
                            (self.distance[j+1]-self.distance[j])
        else:
            slopes = array
        return slopes
    
    def __calculate_FT(self, array_name='Ratio_diff_slope'):
        if array_name == 'Ratio_diff_slope':
            array = self.__Ratio_diff_slope
            array_init = self.__Ratio_obs_slope
        elif array_name == 'Sr_diff_slope':
            array = self.__Sr_diff_slope
            array_init = self.__Sr_obs_slope
        elif array_name == 'Ratio_diff':
            array = self.__Ratio_diff
            array_init = self.__Ratio_obs
        elif array_name == 'Sr_diff':
            array = self.__Sr_diff
            array_init = self.__Sr_obs
        array = array[:,self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
        array_init = array_init[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
        dims = array.shape
        if self.__windowed:
            win = tukey(dims[1], alpha=self.__tukey_factor)
        else:
            win = np.ones(dims[1])
        if not ('slope' in array_name):
            if 'Ratio' in array_name:
                array = array - 1
                array_init = array_init - 1
            else:
                array = array - self.Sr_av
                array_init = array_init - self.Sr_av
        array_win = array*win
        array_win_init = array_init*win
        array = array*win
        array_init = array_init*win
        if self.__odd_extend:
            array_odd = -np.flip(array, axis=1)
            array = np.concatenate((array_odd,array), axis=1)
            array_init_odd = -np.flip(array_init)
            array_init = np.concatenate((array_init_odd,array_init))
        ffts = np.zeros((dims[0],len(self.__freq)),dtype=np.complex_)
        for i in range(dims[0]):
            a = np.fft.rfft(array[i,:])
            ffts[i,:] = a
        ffts_init = np.fft.rfft(array_init)
        mag_ffts = np.abs(ffts)
        mag_ffts_init = np.abs(ffts_init)
        if array_name == 'Ratio_diff':
                array_win = array_win + 1
                array_win_init = array_win_init + 1
        elif array_name == 'Sr_diff':
                array_win = array_win + self.Sr_av
                array_win_init = array_win_init + self.Sr_av
        output = {'FT': ffts, 'FT_i': ffts_init, 'Mag_FT': mag_ffts, 'Mag_FT_i': mag_ffts_init,\
                  'Prof_win': array_win, 'Prof_win_i': array_win_init}
        return output
        
    def __calculate_IFT(self, array_name='Ratio_diff_slope'):
        if array_name == 'Ratio_diff_slope':
            dict_FT = self.__Ratio_slope_FT
        elif array_name == 'Sr_diff_slope':
            dict_FT = self.__Sr_slope_FT
        elif array_name == 'Ratio_diff':
            dict_FT = self.__Ratio_FT
        elif array_name == 'Sr_diff':
            dict_FT = self.__Sr_FT 
        dims = [dict_FT['FT'].shape[0], len(self.__freq), len(self.dist_used)]
        iffts_all = np.zeros(dims)
        iffts_init = np.zeros(dims[1:3])
        amps_all = np.zeros(dims[0:2])
        amps_init = np.zeros(dims[1])
        for j in range(dims[1]):
            for i in range(dims[0]):
                ffts = np.zeros(dims[1],dtype=np.complex_)
                ffts[j] = dict_FT['FT'][i,j]
                iffts = np.fft.irfft(ffts)
                amps_all[i,j] = (max(iffts)-min(iffts))/2
                if self.__odd_extend:
                    iffts = iffts[-dims[2]:]
                iffts_all[i,j,:] = iffts
            ffts_i = np.zeros(dims[1],dtype=np.complex_)
            ffts_i[j] = dict_FT['FT_i'][j]
            iffts_i = np.transpose(np.fft.irfft(ffts_i))
            amps_init[j] = (max(iffts_i)-min(iffts_i))/2
            if self.__odd_extend:
                iffts_i = iffts_i[-dims[2]:]
            iffts_init[j,:] = iffts_i
        output = {'IFT': iffts_all, 'IFT_i': iffts_init, 'Amp_IFT': amps_all, 'Amp_IFT_i': amps_init}
        return output
    
    def __sum_IFT(self, IFT):
        if len(IFT.shape) == 3:
            IFT2 = IFT[:,self.__freq_filt,:]
            sum_IFT = np.nansum(IFT2, axis=1)
        else:
            IFT2 = IFT[self.__freq_filt,:]
            sum_IFT = np.nansum(IFT2, axis=0)
        return sum_IFT
    
    def filt_amps(self):
        l = len(self.__Ratio_slope_IFT['Amp_IFT'][0,:])
        t = len(self.__time)
        idxs = np.zeros(l,dtype=int)
        amps = self.__Ratio_slope_IFT['Amp_IFT']
        time = self.__time
        for f in range(l):
            for i in range(t-3):
                slope1 = (amps[i+1,f]-amps[i,f])/(time[i+1]-time[i])
                slope2 = (amps[i+2,f]-amps[i+1,f])/(time[i+2]-time[i+1])
                slope3 = (amps[i+3,f]-amps[i+2,f])/(time[i+3]-time[i+2])
                # ds1 = (slope2-slope1)/(time[i+1]-time[i])
                # ds2 = (slope3-slope2)/(time[i+2]-time[i+1])
                # or ds2 > ds1
                if slope1 >= 0 or slope2 >= 0 or ((slope2-slope1 > abs(slope1*0.15))\
                                                  and (slope3-slope2 > abs(slope2*0.15))):
                    break
            idxs[f] = i
        return idxs
    
    def IFT_Amp_regress(self, array_name='Ratio_diff_slope'):
        if array_name == 'Ratio_diff_slope':
            dict_IFT = self.__Ratio_slope_IFT
        elif array_name == 'Sr_diff_slope':
            dict_IFT = self.__Sr_slope_IFT
        elif array_name == 'Ratio_diff':
            dict_IFT = self.__Ratio_IFT
        elif array_name == 'Sr_diff':
            dict_IFT = self.__Sr_IFT
        om = 10**math.floor(math.log(self.__time[-1],10))
        def exp_fit(x, a, b, c):
            om = 10**math.floor(math.log(self.__time[-1],10))
            y = a*np.exp(b*x/om+c)
            return y
        iffts_init = dict_IFT['IFT_i']
        iffts_norm = np.zeros(iffts_init.shape)
        amps_init = dict_IFT['Amp_IFT_i']
        amps = dict_IFT['Amp_IFT']
        dim = len(amps_init)
        fits = [[] for i in range(dim)]
        idx_filt = self.filt_amps()
        for i in range(dim):
            if amps_init[i] != 0:
                iffts_norm[i,:] = iffts_init[i,:]/amps_init[i]
            else:
                iffts_norm[i,:] = np.array([0 for j in range(iffts_init.shape[1])])
            if idx_filt[i] != 0:
                try:
                    fits[i] = curve_fit(exp_fit, self.__time[0:idx_filt[i]], amps[0:idx_filt[i],i], p0=[1e-3,-50,1e-1], maxfev=int(5e4))
                except:
                    fits[i] = []
                    print(self.__freq[i])
            else:
                fits[i] = []
        output = {'IFT_norm': iffts_norm, 'IFT_fit': fits, 'Time_om': om}
        return output
    
    def IFT_regress_eval(self, IFT_regress, time_vec):
        if str(type(time_vec)) in ["<class 'float'>", "<class 'int'>", "<class 'numpy.float64'>"]:
            time_vec = np.array([time_vec])
        dims = [len(time_vec), IFT_regress['IFT_norm'].shape[0], IFT_regress['IFT_norm'].shape[1]]
        iffts_regress = np.zeros(dims)
        amp_regress = np.zeros(dims[0:2])
        # iffts_total = np.zeros((dims[0], dims[2]))
        om = IFT_regress['Time_om']
        for f in range(dims[1]):
            fit = IFT_regress['IFT_fit'][f]
            
            for t in range(dims[0]):
                if fit:
                    amp_regress[t,f] = fit[0][0]*np.exp(fit[0][1]*time_vec[t]/om+fit[0][2])
                    iffts_regress[t,f,:] = IFT_regress['IFT_norm'][f,:]*amp_regress[t,f]
                else:
                    amp_regress[t,f] = np.nan
                    iffts_regress[t,f,:] = np.array([np.nan for i in range(dims[2])])
            # if self.__freq_filt[f]:
            #     iffts_total = iffts_total + iffts_regress[:,f,:]
        
        return iffts_regress, amp_regress
                
    def __IFT_regress(self, IFT, IFT_reg):
        ifft_reg, amp_reg = self.IFT_regress_eval(IFT_reg, self.__time)
        dims = amp_reg.shape
        ifft_res = np.zeros(dims)
        for f in range(dims[1]):
            if amp_reg[1,f]:
                ifft_res[:,f] = IFT['Amp_IFT'][:,f]-amp_reg[:,f]
            else:
                ifft_res[:,f] = np.array([np.nan for i in range(dims[0])])
        output = {'IFT_norm': IFT_reg['IFT_norm'], 'IFT_fit': IFT_reg['IFT_fit'], 'Time_om': IFT_reg['Time_om'],\
                  'IFT_reg': ifft_reg, 'Amp_IFT_reg': amp_reg, 'Amp_IFT_res': ifft_res}
        return output
    
    def integrate_Ratio_slope(self, h):
        if len(h.shape) == 2:
            h = h[0,:]
        dim = len(h)
        f0 = np.zeros(dim)
        Sr0_tot = 0
        for i in range(1,dim):
            f0[i] = (f0[i-1] + (self.dist_used[i]-self.dist_used[i-1])*h[i-1])
        f0 = f0*self.__Sr_eq[self.dist_used_idx[0]:self.dist_used_idx[1]+1]
        for i in range(1,dim):
            Sr0_tot = Sr0_tot +  (self.dist_used[i]-self.dist_used[i-1])*f0[i]
        C = 1 - Sr0_tot/(self.Sr_av_used*(self.dist_used[-1]-self.dist_used[0]))
        f = f0+C*self.__Sr_eq[self.dist_used_idx[0]:self.dist_used_idx[1]+1]
        return f
            
    
    def __init__(self, file_path):
        self.__file = os.path.basename(file_path)
        self.__folder = file_path.replace(self.__file,'')
        self.__name = self.__file.replace('_diffused.xlsx', '')
        self.__path_exists = os.path.exists(file_path)
        
        if self.__path_exists:
            os.chdir(self.__folder)
            excel_file = pd.ExcelFile(f'{self.file}')
            input_1 = pd.read_excel(excel_file, 'Transformed SCAPS Data')
            input_2 = pd.read_excel(excel_file,'Diffusion Data - Conc')
            input_3 = pd.read_excel(excel_file,'Diffusion Data - Ratios')
        else:
            input_1 = np.empty(0)
            input_2 = np.empty(0)
            input_3 = np.empty(0)
            print('Warning: No profile loaded')
            
        if isinstance(input_1, pd.DataFrame):
            self.__distance = np.array(input_1['Distance (um)'])
            self.__An = np.array(input_1['An#'])
            self.__An_std = np.array(input_1['An# 1-Sigma'])
            self.__Sr_obs = np.array(input_1['Sr (ug/g)'])
            self.__Sr_obs_std = np.array(input_1['Sr 1-Sigma'])
            self.__Sr_eq = np.array(input_1['Eq. Sr (ug/g)'])
            self.__Sr_eq_std = np.array(input_1['Eq. Sr 1-Sigma'])
            self.__Ratio_obs = np.array(input_1['Sr Ratio (Obs./Eq.)'])
            self.__Ratio_obs_std = np.array(input_1['Ratio 1-Sigma'])
            self.__time = np.array(input_2['Time'])
            Sr_diff = np.delete(np.array(input_2),0,1)
            self.__Sr_diff = np.delete(Sr_diff, np.s_[-1:], axis=1)
            self.__Ratio_diff = np.delete(np.array(input_3),0,1)
        else:
            self.__distance = input_1
            self.__An = input_1
            self.__An_std = input_1
            self.__Sr_obs = input_1
            self.__Sr_obs_std = input_1
            self.__Sr_eq = input_1
            self.__Sr_eq_std = input_1
            self.__Ratio_obs = input_1
            self.__Ratio_obs_std = input_1
            self.__time = input_2
            self.__Sr_diff = input_2
            self.__Ratio_diff = input_3
        
        self.__An_slope = self.__derivate_1D(self.__An)
        self.__Sr_obs_slope =  self.__derivate_1D(self.__Sr_obs)
        self.__Sr_eq_slope =  self.__derivate_1D(self.__Sr_eq)
        self.__Ratio_obs_slope =  self.__derivate_1D(self.__Ratio_obs)
        self.__Sr_diff_slope =  self.__derivate_2D(self.__Sr_diff)
        self.__Ratio_diff_slope =  self.__derivate_2D(self.__Ratio_diff)
        
        self.__dist_sub_idx = [0, len(self.distance)-1]
        self.__dist_used_idx = self.__dist_sub_idx
        self.__windowed = False
        self.__tukey_factor = 0.1
        self.__odd_extend = True
        self.__FT_calculated = False
        self.__correct_reg_init = True
        self.__time_select_idx = 1
        self.__time_val = -3600.0 # time value (seconds) for model prediction
        
        if self.__odd_extend:
            self.__freq = np.fft.rfftfreq(2*len(self.dist_used),d=self.dist_step)
        else:
            self.__freq = np.fft.rfftfreq(len(self.dist_used),d=self.dist_step)
        
        self.__freq_filt = np.array([True for i in self.__freq])
        self.__freq_sub_idx = [0,len(self.__freq)-1]
        self.__mag_filt = 0
        self.__filter_type = 'Freq. filt.'
        self.__Sr_FT = []
        self.__Ratio_FT = []
        self.__Sr_slope_FT = []
        self.__Ratio_slope_FT = []
        
        self.__Sr_IFT = []
        self.__Ratio_IFT = []
        self.__Sr_slope_IFT = []
        self.__Ratio_slope_IFT = []
        
        self.__Ratio_slope_regress = []
        
        self.__fig = []
        
        self.__selected_array = 'Sr_diff'
        self.__plot1_textbox = []
        self.__plot1_ymin = None
        self.__plot1_ymax = None
        self.__plot1_ydelta = None
        self.__plot1_ax = []
        self.__plot1_ylabel = ''
        self.__plot1 = []
        self.__plot1_init = []
        self.__plot1_ydata = []
        self.__plot1_ydata_init = []
        self.__plot1_lims = [[], []]
        
        self.__plot2_ax = []
        self.__plot2 = []
        self.__plot3_ax = []
        self.__plot3 = []
        self.__plot3_2 = []
        self.__plot2_lims = []
        
        self.__plot4_ax = []
        self.__plot4 = []
        self.__plot4_zdata = []
        self.__plot4_lim = []
        self.__plot4_FT = []
        self.__plot4_cbar_ax = []
        self.__plot4_cbar = []
        
        self.__plot5_ax = []
        self.__plot5 = []
        self.__plot5_init = []
        self.__plot5_ymax = []
        self.__plot5_ydelta = []
        self.__plot5_textbox = []
        self.__plot5_lims =[[], [], []]
        self.__plot5_cmap = np.array(['k' for i in self.__freq])
        self.__plot5_scatter = []
        
        self.__plot6_ax = []
        self.__plot6 = []
        self.__plot6_init = []
        self.__plot6_2 = []
        
        self.__plot7_ax = []
        self.__plot7 = []
        self.__plot7_cmap = []
        self.__plot7_lim = []
        
        self.__plot8_ax = []
        self.__plot8 = []
        
        self.__plot9_ax = []
        self.__plot9 = []
        
        self.__plot10_ax = []
        self.__plot10 = []
        self.__plot10_cmap = []
        self.__plot10_lim = []
        self.__plot10_type = []
        
        self.__plot11_ax = []
        self.__plot12_ax = []
    
    @property
    def not_regress(self):
        dim = len(self.__freq)
        if self.__Ratio_slope_regress:
            output = np.full(dim, True, dtype=bool)
            for i in range(dim):
                if np.isnan(self.__Ratio_slope_regress['Amp_IFT_reg'][0,i]):
                    output[i] = False
                else:
                    output[i] = True
        else:
            output = np.array([True for i in range(dim)])
        return output
    
    @property
    def file(self):
        return self.__file
    @property
    def folder(self):
        return self.__folder
    @property
    def name(self):
        return self.__name
    @property
    def path_exists(self):
        return self.__path_exists

    @property
    def distance(self):
        return self.__distance
    @property
    def An(self):
        return self.__An
    @property
    def An_std(self):
        return self.__An_std
    @property
    def Sr_obs(self):
        return self.__Sr_obs
    @property
    def Sr_obs_std(self):
        return self.__Sr_obs_std
    @property
    def Sr_eq(self):
        return self.__Sr_eq 
    @property
    def Sr_eq_std(self):
        return self.__Sr_eq_std
    @property
    def Ratio_obs(self):
        return self.__Ratio_obs
    @property
    def Ratio_obs_std(self):
        return self.__Ratio_obs_std
    @property
    def time(self):
        return self.__time
    @property
    def Sr_diff(self):
        return self.__Sr_diff
    @property
    def Ratio_diff(self):
        return self.__Ratio_diff
    
    @property
    def An_slope(self):
        return self.__An_slope
    @property
    def Sr_obs_slope(self):
        return self.__Sr_obs_slope
    @property
    def Sr_eq_slope(self):
        return self.__Sr_eq_slope
    @property
    def Ratio_obs_slope(self):
        return self.__Ratio_obs_slope
    @property
    def Sr_diff_slope(self):
        return self.__Sr_diff_slope
    @property
    def Ratio_diff_slope(self):
        return self.__Ratio_diff_slope

    @property
    def Sr_av_used(self):
        return np.average(self.__Sr_obs[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1])
    @property
    def Sr_av(self):
        return np.average(self.__Sr_obs)
    
    @property
    def dist_used_idx(self):
        return self.__dist_used_idx
    @dist_used_idx.setter
    def dist_used_idx(self, idx):
        if idx[0] < 0:
            idx[0] = 0
        if idx[1] > len(self.__distance)-1:
            idx[1] = len(self.__distance)-1
        self.__dist_used_idx = idx
        
    @property
    def dist_sub_idx(self):
        return self.__dist_sub_idx
    @dist_sub_idx.setter
    def dist_sub_idx(self, idx):
        if idx[0] < 0:
            idx[0] = 0
        if idx[1] > len(self.__distance)-1:
            idx[1] = len(self.__distance)-1
        if self.__plot1:
            self.__plot1_lims[0][0].set_xdata([self.__distance[idx[0]], self.__distance[idx[0]]])
            self.__plot1_lims[1][0].set_xdata([self.__distance[idx[1]], self.__distance[idx[1]]])
        if self.__plot2:
            self.__plot2_lims[0][0].set_xdata([self.__distance[idx[0]], self.__distance[idx[0]]])
            self.__plot2_lims[1][0].set_xdata([self.__distance[idx[1]], self.__distance[idx[1]]])
        # if self.__plot3:
        #     xdata2 = self.__An[idx[0]:idx[1]+1]
        #     if self.__time_select_idx == 0:
        #         self.__plot3[2][0].set_data(xdata2, self.__Sr_obs[idx[0]:idx[1]+1])
        #     else:
        #         self.__plot3[2][0].set_data(xdata2, self.__Sr_diff[self.__time_select_idx-1,idx[0]:idx[1]+1])
        self.__dist_sub_idx = idx
        
    @property
    def time_select_idx(self):
        return self.__time_select_idx
    @time_select_idx.setter
    def time_select_idx(self, idx):
        if idx < 0:
            Warning('Time index is negative. Index assigned to 0.')
            idx = 0
        if idx > len(self.__time):
            Warning('Time index above maximum. Index assigned to maximum.')
            idx = len(self.__distance)
        self.__time_select_idx = idx
        if self.__plot1:
            if idx == 0:
                self.__plot1[0].set_ydata(self.__plot1_ydata_init)
            else:
                self.__plot1[0].set_ydata(self.__plot1_ydata[idx-1,:])
            self.__plot1_ax.set_ylabel(self.__plot1_ylabel)
            self.__plot1_ymin = np.min(self.__plot1_ydata)
            self.__plot1_ymax = np.max(self.__plot1_ydata)
            self.__plot1_ydelta = (self.__plot1_ymax-self.__plot1_ymin)/20
            self.__plot1_ax.set_ylim(self.__plot1_ymin-self.__plot1_ydelta, self.__plot1_ymax+self.__plot1_ydelta)
            self.__plot1_textbox.set_text(self.time_str())
        if self.__plot3:
            if idx == 0:
                self.__plot3[2][0].set_ydata(self.__Sr_obs)
            else:
                self.__plot3[2][0].set_ydata(self.__Sr_diff[idx-1,:])
            self.__plot3_textbox.set_text(self.time_str())
        if self.__plot4:
            if idx == 0:
                self.__plot4_lim[0].set_xdata([self.__time[0]-1, self.__time[0]-1])
            else:
                self.__plot4_lim[0].set_xdata([self.__time[idx-1], self.__time[idx-1]])
        if self.__plot5:
            if idx == 0:
                self.__plot5[0].set_ydata(self.__plot4_FT['Mag_FT_i'])
                self.__plot5_scatter.remove()
                self.__plot5_scatter = self.__plot5_ax.scatter(self.__freq, self.__plot4_FT['Mag_FT_i'], 20, self.__plot5_cmap, picker=True, edgecolor='k', zorder=10)
            else:
                self.__plot5[0].set_ydata(self.__plot4_FT['Mag_FT'][idx-1,:])
                self.__plot5_scatter.remove()
                self.__plot5_scatter = self.__plot5_ax.scatter(self.__freq, self.__plot4_FT['Mag_FT'][idx-1,:], 20, self.__plot5_cmap, picker=True, edgecolor='k', zorder=10)
            self.__plot5_textbox.set_text(self.time_str())
            self.mark_not_regress()
        if self.__plot6:
            ydata2 = self.__sum_IFT(self.__plot6_IFT['IFT'])
            ydata2_init = self.__sum_IFT(self.__plot6_IFT['IFT_i'])
            if idx == 0:
                self.__plot6[0][0].set_ydata(self.__plot4_FT['Prof_win_i'])
                self.__plot6[1][0].set_ydata(ydata2_init)
            else:
                self.__plot6[0][0].set_ydata(self.__plot4_FT['Prof_win'][idx-1,:])
                self.__plot6[1][0].set_ydata(ydata2[idx-1,:])
        if self.__plot6_2:
            ydata = self.__sum_IFT(self.__Ratio_slope_regress['IFT_reg'])
            if idx == 0:
                self.__plot6_2[0].set_ydata(ydata[idx,:])
            else:
                self.__plot6_2[0].set_ydata(ydata[idx-1,:])
        if self.__plot7:
            self.__plot7_lim[0].set_xdata([self.time_select, self.time_select])
            for i in range(len(self.__freq_filt)):
                if self.__freq_filt[i]:
                    self.__plot7[i][0].set_linestyle('-')
                else:
                    self.__plot7[i][0].set_linestyle('None')
        if self.__plot10:
            self.__plot10_lim[0].set_xdata([self.time_select, self.time_select])
            for i in range(len(self.__freq_filt)):
                if self.__freq_filt[i]:
                    self.__plot10[i][0].set_linestyle('-')
                else:
                    self.__plot10[i][0].set_linestyle('None')
    @property
    def dist_sub(self):
        return self.distance[self.__dist_sub_idx[0]:self.__dist_sub_idx[1]+1]
    @property
    def dist_used(self):
        return self.distance[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
    
    @property
    def time_select(self):
        if self.__time_select_idx == 0:
            return 0
        else:
            return self.__time[self.__time_select_idx-1]
    @property
    def dist_step(self):
        return self.__distance[-1]/(self.__distance.shape[0]-1)
    
    @property
    def windowed(self):
        return self.__windowed
    @windowed.setter
    def windowed(self, val):
        if isinstance(val, bool):
            self.__windowed = val
        else:
            Warning('value assigned to windowed variable is not a boolean. False was assign instead.')
            self.__windowed = False
    @property
    def tukey_factor(self):
        return self.__tukey_factor
    @tukey_factor.setter
    def tukey_factor(self, val):
        if val < 0:
            val = 0
        elif val > 1:
            val = 1
        self.__tukey_factor = val
                    
    @property
    def correct_reg_init(self):
        return self.__correct_reg_init
    @correct_reg_init.setter
    def correct_reg_init(self, val):
        if isinstance(val, bool):
            self.__correct_reg_init = val
        else:
            Warning('value assigned to correction variable is not a boolean. False was assign instead.')
            self.__correct_reg_init = False
        if self.__plot8:
            self.time_val = self.__time_val
            
    @property
    def odd_extend(self):
        return self.__odd_extend
    @odd_extend.setter
    def odd_extend(self, val):
        if isinstance(val, bool):
            self.__odd_extend = val
            if val:
                self.__freq = np.fft.rfftfreq(2*len(self.dist_used),d=self.dist_step)
            else:
                self.__freq = np.fft.rfftfreq(len(self.dist_used),d=self.dist_step)
        else:
            Warning('value assigned to odd_extend variable is not a boolean. False was assign instead.')
            self.__odd_extend = False
            self.__freq = np.fft.rfftfreq(len(self.distance),d=self.dist_step)
            
    @property
    def freq(self):
        return self.__freq
    @property
    def freq_filt(self):
        return self.__freq_filt
    def update_freq_filt(self):
        self.__freq_filt = np.array([True for i in self.__freq])
        if self.__filter_type == 'Freq. filt.':
            for i in range(len(self.__freq_filt)):
                if i < self.__freq_sub_idx[0] or i > self.__freq_sub_idx[1]:
                    self.__freq_filt[i] = False
                    self.__plot5_cmap[i] = 'r'
                else:
                    self.__freq_filt[i] = True
                    self.__plot5_cmap[i] = 'k'
        elif self.__filter_type == 'Mag. thres.':
            if self.__plot4:
                for i in range(len(self.__freq_filt)):
                    if self.__plot4_FT['Mag_FT_i'][i] >= self.__mag_filt:
                        self.__freq_filt[i] = True
                        self.__plot5_cmap[i] = 'k'
                    else:
                        self.__freq_filt[i] = False
                        self.__plot5_cmap[i] = 'r'
        elif self.__filter_type == 'Freq. select':
            for i in range(len(self.__freq_filt)):
                if self.__plot5_cmap[i] == 'k':
                    self.__freq_filt[i] = True
                else:
                    self.__freq_filt[i] = False
        self.time_select_idx = self.__time_select_idx
        self.mark_not_regress()
        if self.__plot8:
            self.time_val = self.__time_val
    @property
    def freq_sub_idx(self):
        return self.__freq_sub_idx
    @freq_sub_idx.setter
    def freq_sub_idx(self, idx):
        if idx[0] < 0:
            Warning('Frequency index is negative. Index assigned to 0.')
            idx[0] = 0
        if idx[1] > len(self.__freq)-1:
            Warning('Frequency index above maximum. Index assigned to maximum.')
            idx[1] = len(self.__freq)-1
        if idx[0] > idx[1]:
            Warning('Frequency index 1 is higher than index 2 above maximum. Values inverted.')
            idx = [idx[1], idx[0]]
        self.__freq_sub_idx = idx
        if self.__plot5:
            self.__plot5_lims[0][0].set_xdata([self.freq_sub[0], self.freq_sub[0]])
            self.__plot5_lims[1][0].set_xdata([self.freq_sub[-1], self.freq_sub[-1]])
            self.update_freq_filt() 
        
    @property
    def freq_sub(self):
        return self.__freq[self.__freq_sub_idx[0]:self.__freq_sub_idx[1]+1]
    @property
    def mag_filt(self):
        return self.__mag_filt
    @mag_filt.setter
    def mag_filt(self, val):
        if val < 0:
            val = 0
        self.__mag_filt = val
        if self.__plot5:
            self.__plot5_lims[2][0].set_ydata([self.mag_filt, self.mag_filt])
            self.update_freq_filt()
    
    @property
    def filter_type(self):
        return self.__filter_type
    @filter_type.setter
    def filter_type(self, str_filt):
        if str_filt in ['Freq. filt.', 'Mag. thres.', 'Freq. select']:
            self.__filter_type = str_filt
            self.__plot5_cmap = np.array(['k' for i in self.__plot5_cmap])
            ydata = self.__plot5_scatter.get_offsets()[:,1]
            self.__plot5_scatter.remove()
            self.__plot5_scatter = self.__plot5_ax.scatter(self.__freq, ydata, 20, self.__plot5_cmap, picker=True)
            self.update_freq_filt()
        else:
            Warning('Wrong filter label. Not updated')
            self.__filter_type = self.__filter_type
    @property
    def Sr_FT(self):
        return self.__Sr_FT   
    @property
    def Ratio_FT(self):
        return self.__Ratio_FT
    @property
    def Sr_slope_FT(self):
        return self.__Sr_slope_FT
    @property
    def Ratio_slope_FT(self):
        return self.__Ratio_slope_FT
    @property
    def Sr_IFT(self):
        return self.__Sr_IFT
    @property
    def Ratio_IFT(self):
        return self.__Ratio_IFT
    @property
    def Sr_slope_IFT(self):
        return self.__Sr_slope_IFT
    @property
    def Ratio_slope_IFT(self):
        return self.__Ratio_slope_IFT
    @property
    def FT_calculated(self):
        return self.__FT_calculated
    @property
    def Ratio_slope_regress(self):
        return self.__Ratio_slope_regress
    
    def Calculate_FTs(self):
        self.__dist_used_idx = self.__dist_sub_idx
        if self.__odd_extend:
            self.__freq = np.fft.rfftfreq(2*len(self.dist_used),d=self.dist_step)
        else:
            self.__freq = np.fft.rfftfreq(len(self.dist_used),d=self.dist_step)
        self.__freq_filt = np.array([True for i in self.__freq])
        # self.__Sr_FT = self.__calculate_FT('Sr_diff')
        # self.__Ratio_FT = self.__calculate_FT('Ratio_diff')
        # self.__Sr_slope_FT = self.__calculate_FT('Sr_diff_slope')
        self.__Ratio_slope_FT = self.__calculate_FT('Ratio_diff_slope')
        # self.__Sr_IFT = self.__calculate_IFT('Sr_diff')
        # self.__Ratio_IFT = self.__calculate_IFT('Ratio_diff')
        # self.__Sr_slope_IFT = self.__calculate_IFT('Sr_diff_slope')
        self.__Ratio_slope_IFT = self.__calculate_IFT('Ratio_diff_slope')
        self.__FT_calculated = True
        
    def regress_IFTs(self, val = 'calc'):
        if val == 'calc':
            regress = self.IFT_Amp_regress()
            self.__Ratio_slope_regress = self.__IFT_regress(self.__Ratio_slope_IFT, regress)
        elif val == 'reset':
            self.__Ratio_slope_regress = []
            
    @property
    def selected_array(self):
        return self.__selected_array
    @selected_array.setter
    def selected_array(self, str_arr):
        if str_arr in ['Ratio_diff_slope', 'Ratio_diff', 'Sr_diff_slope', 'Sr_diff']:
            self.__plot4_FT = self.__Ratio_slope_FT
            self.__plot6_IFT = self.__Ratio_slope_IFT
            if str_arr == 'Ratio_diff_slope':
                self.__plot1_ydata = self.__Ratio_diff_slope
                self.__plot1_ydata_init = self.__Ratio_obs_slope
                self.__plot1_ylabel = '$Sr_{Obs}$/$Sr_{Eq}$ slope (1/$\mu$m)'
            elif str_arr == 'Sr_diff_slope':
                self.__plot1_ydata = self.__Sr_diff_slope
                self.__plot1_ydata_init = self.__Sr_obs_slope
                self.__plot1_ylabel = '$Sr_{Obs}$ slope (ppm/$\mu$m)'
            elif str_arr == 'Ratio_diff':
                self.__plot1_ydata = self.__Ratio_diff
                self.__plot1_ydata_init = self.__Ratio_obs
                self.__plot1_ylabel = '$Sr_{Obs}$/$Sr_{Eq}$'
            elif str_arr == 'Sr_diff':
                self.__plot1_ydata = self.__Sr_diff
                self.__plot1_ydata_init = self.__Sr_obs
                self.__plot1_ylabel = '$Sr_{Obs}$ (ppm)'
            if self.__plot1:
                self.time_select_idx = self.__time_select_idx
                self.__plot1_init[0].set_ydata(self.__plot1_ydata_init)
                self.__plot1_lims[0][0].set_ydata(self.__plot1_ax.get_ylim())
                self.__plot1_lims[1][0].set_ydata(self.__plot1_ax.get_ylim())
            self.__selected_array = str_arr
            
        else:
            Warning('Array name not available. The array name was not changed')
            self.__selected_array = self.__selected_array
            
    def time_str(self, t_type=0):
        if t_type == 0:
            t = self.time_select
        elif t_type == 1:
            t = -self.__time_val
        if t < 60:
            ts = str(round(t)) + " secs"
        elif t < 60*60:
            ts = str(round(t/60, 2)) + " mins"
        elif t < 60*60*24:
            ts = str(round(t/(60*60), 2)) + " hrs"
        elif t < 60*60*24*(365/12):
            ts = str(round(t/(60*60*24), 2)) + " days"
        elif t < 60*60*24*365:
            ts = str(round(t/(60*60*24*(365/12)), 2)) + " months"
        elif t < 60*60*24*365*1000:
            ts = str(round(t/(60*60*24*365), 2)) + " yrs"
        elif t < 60*60*24*365*1000*1000:
            ts = str(round(t/(60*60*24*365*1000), 2)) + " kyrs"
        else:
            ts = str(round(t/(60*60*24*365*1000*1000), 2)) + " Myrs"
        if t_type == 1:
            ts = '-'+ts
        return ts
    
    @property
    def time_val(self):
        return self.__time_val
    @time_val.setter
    def time_val(self, val):
        self.__time_val = val
        if self.__plot8:
            iffts, amps = self.IFT_regress_eval(self.__Ratio_slope_regress, self.__time_val)
            ydata = self.__sum_IFT(iffts)
            self.__plot8[1][0].set_ydata(ydata[0,:])
            self.__plot8_ymin = np.min(ydata)
            self.__plot8_ymax = np.max(ydata)
            self.__plot8_ydelta = (self.__plot8_ymax-self.__plot8_ymin)/20
            self.__plot8_ax.set_ylim(self.__plot8_ymin-self.__plot8_ydelta, self.__plot8_ymax+self.__plot8_ydelta)
        if self.__plot9:
            iffts, amps = self.IFT_regress_eval(self.__Ratio_slope_regress, self.__time_val)
            ydata = self.__sum_IFT(iffts)
            ydata2 = self.integrate_Ratio_slope(ydata)
            if self.__correct_reg_init:
                iffts_i, amps_i = self.IFT_regress_eval(self.__Ratio_slope_regress, 0)
                ydata_i = self.__sum_IFT(iffts_i)
                ydata2_i = self.integrate_Ratio_slope(ydata_i)
                ydata2 = ydata2-ydata2_i+self.__Sr_obs[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
            self.__plot9[1][0].set_ydata(ydata2)
            self.__plot9_ymin = np.min(ydata2)
            self.__plot9_ymax = np.max(ydata2)
            self.__plot9_ydelta = (self.__plot9_ymax-self.__plot9_ymin)/20
            self.__plot9_ax.set_ylim(self.__plot9_ymin-self.__plot9_ydelta, self.__plot9_ymax+self.__plot9_ydelta)
            self.__plot3_2[0].set_ydata(ydata2)
        if self.__plot3_2:
            self.__plot3_2_textbox.set_text(self.time_str(1))
            
    @property
    def fig(self):
        return self.__fig
    @fig.setter
    def fig(self, fig):
        self.__fig = fig
    @property
    def plot1_ax(self):
        return self.__plot1_ax
    @plot1_ax.setter
    def plot1_ax(self, ax):
        self.__plot1_ax = ax
    @property
    def plot1(self):
        return self.__plot1
    @property
    def plot1_lims(self):
        return self.__plot1_lims
    
    def plot1_func(self):
        self.__plot1_ax.clear()
        self.selected_array = self.__selected_array
        self.__plot1_ax.set_ylabel(self.__plot1_ylabel)
        self.__plot1_ax.set_xlabel(r'Distance ($\mu$m)')
        self.__plot1_init = self.__plot1_ax.plot(self.distance, self.__plot1_ydata_init, 'k--')
        if self.__time_select_idx == 0:
            self.__plot1 = self.__plot1_ax.plot(self.distance, self.__plot1_ydata_init, 'k-', marker='.')
        else:
            self.__plot1 = self.__plot1_ax.plot(self.distance, self.__plot1_ydata[self.__time_select_idx-1,:], 'k-', marker='.')
        self.__plot1_ax.set_xlim(self.distance[0], self.distance[-1])
        self.__plot1_ymin = np.min(self.__plot1_ydata)
        self.__plot1_ymax = np.max(self.__plot1_ydata)
        self.__plot1_ydelta = (self.__plot1_ymax-self.__plot1_ymin)/20
        self.__plot1_ax.set_ylim(self.__plot1_ymin-self.__plot1_ydelta, self.__plot1_ymax+self.__plot1_ydelta)
        self.__plot1_lims[0] = self.__plot1_ax.plot([self.dist_used[0], self.dist_used[0]], self.__plot1_ax.get_ylim(), 'r', linestyle='None') 
        self.__plot1_lims[1] = self.__plot1_ax.plot([self.dist_used[-1], self.dist_used[-1]], self.__plot1_ax.get_ylim(), 'r', linestyle='None') 
        self.__plot1_textbox = self.__plot1_ax.text(0.98, 0.05, self.time_str(), transform=self.__plot1_ax.transAxes, fontsize=11, va='center', ha='right')
    
    @property
    def plot2_ax(self):
        return self.__plot2_ax
    @plot2_ax.setter
    def plot2_ax(self, ax):
        self.__plot2_ax = ax
    @property
    def plot2_ax_twin(self):
        return self.__plot2_ax_twin
    @plot2_ax_twin.setter
    def plot2_ax_twin(self, ax):
        self.__plot2_ax_twin = ax
    @property
    def plot2(self):
        return self.__plot2
    @property
    def plot2_lims(self):
        return self.__plot2_lims
    def plot2_func(self):
        self.__plot2_ax.clear()
        self.__plot2_ax_twin.clear()
        self.__plot2 = [[], [], []]
        self.__plot2[0] = [[], [], []]
        self.__plot2[0][0] = self.__plot2_ax.plot(self.distance, self.__An, '-', marker='.', color='k')
        self.__plot2[0][1] = self.__plot2_ax.plot(self.distance, self.__An-self.__An_std, ':', color='k')
        self.__plot2[0][2] = self.__plot2_ax.plot(self.distance, self.__An+self.__An_std, ':', color='k')
        self.__plot2_ax.set_ylabel('An (mol frac.)')
        self.__plot2_ax.set_xlabel(r'Distance ($\mu$m)')
        self.__plot2_ax.set_xlim(self.distance[0], self.distance[-1])
        self.__plot2_ymin = np.min(self.__An-self.__An_std)
        self.__plot2_ymax = np.max(self.__An+self.__An_std)
        self.__plot2_ydelta = (self.__plot2_ymax-self.__plot2_ymin)/20
        self.__plot2_ax.set_ylim(self.__plot2_ymin-self.__plot2_ydelta, self.__plot2_ymax+self.__plot2_ydelta)
        self.__plot2[1] = [[], [], []]
        self.__plot2[1][0] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_obs, '-', marker='.', color='b')
        self.__plot2[1][1] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_obs-self.__Sr_obs_std, ':', color='b')
        self.__plot2[1][2] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_obs+self.__Sr_obs_std, ':', color='b')
        self.__plot2[2] = [[], [], []]
        self.__plot2[2][0] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_eq, '-', marker='.', color='g')
        self.__plot2[2][1] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_eq-self.__Sr_eq_std, ':', color='g')
        self.__plot2[2][2] = self.__plot2_ax_twin.plot(self.distance, self.__Sr_eq+self.__Sr_eq_std, ':', color='g')
        self.__plot2_ax_twin.set_ylabel('Sr (ppm)')
        self.__plot2_twin_ymin = min([np.min(self.__Sr_obs-self.__Sr_obs_std), np.min(self.__Sr_eq-self.__Sr_eq_std)])
        self.__plot2_twin_ymax = max([np.max(self.__Sr_obs+self.__Sr_obs_std), np.max(self.__Sr_eq+self.__Sr_eq_std)])
        self.__plot2_twin_ydelta = (self.__plot2_twin_ymax-self.__plot2_twin_ymin)/20
        self.__plot2_ax_twin.set_ylim(self.__plot2_twin_ymin-self.__plot2_twin_ydelta, self.__plot2_twin_ymax+self.__plot2_twin_ydelta)
        self.__plot2_lims = [[], []]
        self.__plot2_lims[0] = self.__plot2_ax.plot([self.dist_used[0], self.dist_used[0]], self.__plot2_ax.get_ylim(), 'r', linestyle='None') 
        self.__plot2_lims[1] = self.__plot2_ax.plot([self.dist_used[-1], self.dist_used[-1]], self.__plot2_ax.get_ylim(), 'r', linestyle='None') 
        
    @property
    def plot3_ax(self):
        return self.__plot2_ax
    @plot3_ax.setter
    def plot3_ax(self, ax):
        self.__plot3_ax = ax
    @property
    def plot3(self):
        return self.__plot3
    def plot3_func(self):
        self.__plot3_ax.clear()
        self.__plot3 = [[], [], []]
        self.__plot3[0] = [[], [], []]
        self.__plot3_ax.set_xlabel('An (mol frac.)')
        self.__plot3_ax.set_ylabel('Sr (ppm)')
        idx_sort = np.argsort(self.__An)
        xdata_sort = self.__An[idx_sort]
        xdata1 = self.__An
        xdata2 = self.__An[self.__dist_sub_idx[0]:self.__dist_sub_idx[1]+1]
        ydata_sort = self.__Sr_eq[idx_sort]
        self.__plot3[0][0] = self.__plot3_ax.plot(xdata_sort, ydata_sort, 'k-')
        self.__plot3[0][1] = self.__plot3_ax.plot(xdata_sort, ydata_sort-self.__Sr_eq_std[idx_sort], 'k:')
        self.__plot3[0][2] = self.__plot3_ax.plot(xdata_sort, ydata_sort+self.__Sr_eq_std[idx_sort], 'k:')
        self.__plot3[1] = self.__plot3_ax.errorbar(xdata1, self.__Sr_obs, self.__Sr_obs_std, self.__An_std, 'k.', alpha=0.2)
        if self.__time_select_idx == 0:
            self.__plot3[2] = self.__plot3_ax.plot(xdata2, self.__Sr_obs,'b.')
        else:
            self.__plot3[2] = self.__plot3_ax.plot(xdata2, self.__Sr_diff[self.__time_select_idx-1,:],'b.')
        self.__plot3_ax.set_xlim(self.__plot3_ax.get_xlim())
        self.__plot3_ax.set_ylim(self.__plot3_ax.get_ylim())
        self.__plot3_textbox = self.__plot3_ax.text(0.98, 0.95, self.time_str(), transform=self.__plot3_ax.transAxes, fontsize=11, va='center', ha='right', color='b')
    @property
    def plot3_2(self):
        return self.__plot3_2
    def plot3_2_func(self):
        if self.__plot3:
            iffts, amps = self.IFT_regress_eval(self.__Ratio_slope_regress, self.__time_val)
            ydata = self.__sum_IFT(iffts)
            ydata2 = self.integrate_Ratio_slope(ydata)
            if self.__correct_reg_init:
                iffts_i, amps_i = self.IFT_regress_eval(self.__Ratio_slope_regress, 0)
                ydata_i = self.__sum_IFT(iffts_i)
                ydata2_i = self.integrate_Ratio_slope(ydata_i)
                ydata2 = ydata2-ydata2_i+self.__Sr_obs[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
            xdata = self.__An[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
            self.__plot3_2 = self.__plot3_ax.plot(xdata, ydata2, 'r.')
            self.__plot3_2_textbox = self.__plot3_ax.text(0.98, 0.90, self.time_str(1), transform=self.__plot3_ax.transAxes, fontsize=11, va='center', ha='right', color='r')
    @property
    def plot4_ax(self):
        return self.__plot4_ax
    @plot4_ax.setter
    def plot4_ax(self, ax):
        self.__plot4_ax = ax
    @property
    def plot4(self):
        return self.__plot4
    @property
    def plot4_cbar(self):
        return self.__plot4_cbar
    @plot4_cbar.setter
    def plot4_cbar(self, cbar):
        self.__plot4_cbar = cbar
    @property
    def plot4_cbar_ax(self):
        return self.__plot4_cbar_ax
    @plot4_cbar_ax.setter
    def plot4_cbar_ax(self, cbar):
        self.__plot4_cbar_ax = cbar
    def plot4_func(self):
        self.__plot4_ax.clear()
        if not self.__plot4:
            self.selected_array = self.__selected_array
        self.__plot4_mesh = np.meshgrid(self.__time, self.__freq)
        mags = np.transpose(self.__plot4_FT['Mag_FT'])
        mags_norm = mags/np.max(mags)
        self.__plot4_ax.set_title('FT magnitude spectra over time')
        self.__plot4_ax.set_xlabel('Time (s)')
        self.__plot4_ax.set_ylabel(r'Frequency (1/$\mu$m)')
        self.__plot4 = self.__plot4_ax.contourf(self.__plot4_mesh[0], self.__plot4_mesh[1],\
                        mags_norm, levels = np.linspace(0,1,100), cmap='jet', extend='min')
        if not self.__plot4_cbar:
            self.__plot4_cbar_ax = self.__plot4_ax.inset_axes([0.85, 0.02, 0.2, 0.96])
            self.__plot4_cbar = self.__fig.colorbar(self.__plot4, ax=self.__plot4_cbar_ax, ticks=[0, 1])
            self.__plot4_cbar.set_label("Norm. Magnitude",rotation=270)
            self.__plot4_cbar_ax.set_visible(False)
        self.__plot4_ax.set_xscale('log')
        self.__plot4_ax.set_ylim(0, self.__freq[-1])
        self.__plot4_ax.set_xlim(self.__time[0], self.__time[-1])
        self.__plot4_lim = self.__plot4_ax.plot([self.time_select, self.time_select],self.__plot4_ax.get_ylim(), 'w-')
    
    @property
    def plot5_ax(self):
        return self.__plot5_ax
    @plot5_ax.setter
    def plot5_ax(self, ax):
        self.__plot5_ax = ax
    @property
    def plot5(self):
        return self.__plot5
    @property
    def plot5_lims(self):
        return self.__plot5_lims
    @property
    def plot5_ymax(self):
        return self.__plot5_ymax
    @property
    def plot5_ydelta(self):
        return self.__plot5_ydelta
    def plot5_func(self):
        self.__plot5_ax.clear()
        if self.__time_select_idx == 0:
            ydata = self.__plot4_FT['Mag_FT_i']
        else:
            ydata = self.__plot4_FT['Mag_FT'][self.__time_select_idx-1,:]
        self.__plot5_init = self.__plot5_ax.plot(self.__freq, self.__plot4_FT['Mag_FT_i'], 'k--', zorder=1)
        self.__plot5 = self.__plot5_ax.plot(self.__freq, ydata, 'k-', zorder=2)
        self.__plot5_ax.set_xlim(self.__freq[0], self.__freq[-1])
        self.__plot5_ymax = np.max(self.__plot4_FT['Mag_FT'])
        self.__plot5_ydelta = self.__plot5_ymax/20
        self.__plot5_ax.set_ylim(0, self.__plot5_ymax+self.__plot5_ydelta)
        self.__plot5_ax.set_xlabel(r'Frequency (1/$\mu$m)')
        self.__plot5_ax.set_ylabel('Magnitude')
        self.__plot5_textbox = self.__plot5_ax.text(0.98, 0.95, self.time_str(), transform=self.__plot5_ax.transAxes, fontsize=11, va='center', ha='right')
        self.__plot5_lims[0] = self.__plot5_ax.plot([self.freq_sub[0], self.freq_sub[0]], self.__plot5_ax.get_ylim(), 'r', linestyle='None')
        self.__plot5_lims[1] = self.__plot5_ax.plot([self.freq_sub[-1], self.freq_sub[-1]], self.__plot5_ax.get_ylim(), 'r', linestyle='None')
        self.__plot5_lims[2] = self.__plot5_ax.plot(self.__plot5_ax.get_xlim(), [self.__mag_filt, self.__mag_filt], 'r', linestyle='None')
        categories = np.zeros(len(self.__freq),dtype=int)
        colormap5 = np.array(['k', 'r'])
        self.__plot5_cmap = colormap5[categories]
        self.__plot5_scatter = self.__plot5_ax.scatter(self.__freq, ydata, 20, self.__plot5_cmap, edgecolor='k', picker=True, zorder=10)
        self.fig.canvas.mpl_connect('pick_event', self.onpick1)
        
    def onpick1(self, event):
        ind = event.ind
        if self.__filter_type == 'Freq. select':
            for i in range(len(ind)):
                if self.__plot5_cmap[ind[i]] == 'k':
                    self.__plot5_cmap[ind[i]] = 'r'
                else:
                    self.__plot5_cmap[ind[i]] = 'k'
            self.time_select_idx = self.__time_select_idx
            self.update_freq_filt()
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
    
    def mark_not_regress(self):
        if self.__plot5_scatter:
            dim = len(self.__freq)
            bl = self.not_regress
            cmap = ['k' for i in range(dim)]
            for i in range(dim):
                if not bl[i]:
                    cmap[i] = 'r'
            self.__plot5_scatter.set_edgecolor(cmap)
            
    @property
    def plot6_ax(self):
        return self.__plot6_ax
    @plot6_ax.setter
    def plot6_ax(self, ax):
        self.__plot6_ax = ax
    @property
    def plot6(self):
        return self.__plot6
    @property
    def plot6_init(self):
        return self.__plot6_init
    def plot6_func(self):
        self.__plot6_ax.clear()
        self.__plot6_ax.set_xlabel(r'Distance ($\mu$m)')
        self.__plot6_ax.set_ylabel('$Sr_{Obs}$/$Sr_{Eq}$ slope (1/$\mu$m)')
        self.__plot6_ax.set_xlim([self.dist_used[0], self.dist_used[-1]])
        self.__plot6_init = self.__plot6_ax.plot(self.dist_used, self.__plot4_FT['Prof_win_i'], 'k--')
        self.__plot6_ax.set_ylim(self.__plot6_ax.get_ylim())
        self.__plot6 = [[], []]
        if self.__time_select_idx == 0:
            ydata = self.__plot4_FT['Prof_win_i']
        else:
            ydata = self.__plot4_FT['Prof_win'][self.__time_select_idx-1,:]
        self.__plot6[0] = self.__plot6_ax.plot(self.dist_used, ydata, 'k-', marker='.')
        ydata2 = self.__sum_IFT(self.__plot6_IFT['IFT'])
        ydata2_init = self.__sum_IFT(self.__plot6_IFT['IFT_i'])
        if self.__time_select_idx == 0:
            self.__plot6[1] = self.__plot6_ax.plot(self.dist_used, ydata2_init, 'b-', marker='.')
        else:
            self.__plot6[1] = self.__plot6_ax.plot(self.dist_used, ydata2[self.__time_select_idx-1,:], 'b-', marker='.')
    
    @property
    def plot6_2(self):
        return self.__plot6_2
    def plot6_reg(self):
        if self.__plot6:
            print(self.__selected_array)
            ydata = self.__sum_IFT(self.__Ratio_slope_regress['IFT_reg'])
            if self.__time_select_idx == 0:
                
                self.__plot6_2 = self.__plot6_ax.plot(self.dist_used,ydata[self.__time_select_idx,:], 'r-')
            else:
                self.__plot6_2 = self.__plot6_ax.plot(self.dist_used,ydata[self.__time_select_idx-1,:], 'r-')
                
    @property
    def plot7_ax(self):
        return self.__plot7_ax
    @plot7_ax.setter
    def plot7_ax(self, ax):
        self.__plot7_ax = ax
    @property
    def plot7(self):
        return self.__plot7
    def plot7_func(self):
        self.__plot7_ax.clear()
        self.__plot7_ax.set_xlabel('Time (s)')
        self.__plot7_ax.set_ylabel('Amplitude [' + self.__plot1_ylabel + ']')
        self.__plot7_ax.set_title('Rate of amplitude decrease of each frequency')
        self.__plot7_ax.set_xlim(self.__time[0], self.__time[-1])
        self.__plot7_ax.set_ylim(0, np.max(self.__plot6_IFT['Amp_IFT']))
        dim = len(self.__freq)
        self.__plot7 = [[] for i in range(dim)]
        self.__plot7_cmap = pl.cm.jet(self.__freq)
        for i in range(dim):
            self.__plot7[i] = self.__plot7_ax.plot(self.__time, self.__plot6_IFT['Amp_IFT'][:,i], color=self.__plot7_cmap[i], linestyle = '-')
            if not self.__freq_filt[i]:
                self.__plot7[i][0].set_linestyle('None')
        self.__plot7_lim = self.__plot7_ax.plot([self.time_select, self.time_select], self.__plot7_ax.get_ylim(), 'k-')
        self.__plot7_ax.ticklabel_format(axis='y', style='sci', scilimits = (-2,3))
    
        
    @property
    def plot8_ax(self):
        return self.__plot8_ax
    @plot8_ax.setter
    def plot8_ax(self, ax):
        self.__plot8_ax = ax
    @property
    def plot8(self):
        return self.__plot8
    def plot8_func(self):
        self.__plot8_ax.clear()
        self.__plot8 = [[], []]
        self.__plot8[0] = self.__plot8_ax.plot(self.dist_used, self.__Ratio_slope_FT['Prof_win_i'], 'k--')
        iffts, amps = self.IFT_regress_eval(self.__Ratio_slope_regress, self.__time_val)
        ydata = self.__sum_IFT(iffts)
        self.__plot8[1] = self.__plot8_ax.plot(self.dist_used, ydata[0,:], 'r-', marker='.')
        self.__plot8_ax.set_xlim(self.dist_used[0], self.dist_used[-1])
        self.__plot8_ymin = np.nanmin(ydata)
        self.__plot8_ymax = np.nanmax(ydata)
        self.__plot8_ydelta = (self.__plot8_ymax-self.__plot8_ymin)/20
        self.__plot8_ax.set_ylim(self.__plot8_ymin-self.__plot8_ydelta, self.__plot8_ymax+self.__plot8_ydelta)
        self.__plot8_ax.set_title(r'Modeled $Sr/Sr_{Eq}$ slope profile')
        self.__plot8_ax.set_ylabel('$Sr/Sr_{Eq}$ slope (1/$\mu$m)')
        self.__plot8_ax.set_xlabel(r'Distance ($\mu$m)')
    @property
    def plot9_ax(self):
        return self.__plot9_ax
    @plot9_ax.setter
    def plot9_ax(self, ax):
        self.__plot9_ax = ax
    @property
    def plot9(self):
        return self.__plot9
    def plot9_func(self):
        self.__plot9_ax.clear()
        self.__plot9 = [[], []]
        self.__plot9[0] = self.__plot9_ax.plot(self.dist_used, self.__Sr_obs[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1], 'k--')
        iffts, amps = self.IFT_regress_eval(self.__Ratio_slope_regress, self.__time_val)
        ydata = self.__sum_IFT(iffts)
        ydata2 = self.integrate_Ratio_slope(ydata)
        if self.__correct_reg_init:
            iffts_i, amps_i = self.IFT_regress_eval(self.__Ratio_slope_regress, 0)
            ydata_i = self.__sum_IFT(iffts_i)
            ydata2_i = self.integrate_Ratio_slope(ydata_i)
            ydata2 = ydata2-ydata2_i+self.__Sr_obs[self.__dist_used_idx[0]:self.__dist_used_idx[1]+1]
        self.__plot9[1] = self.__plot9_ax.plot(self.dist_used, ydata2, 'r-', marker='.')
        self.__plot9_ax.set_xlim(self.dist_used[0], self.dist_used[-1])
        self.__plot9_ymin = np.min(ydata2)
        self.__plot9_ymax = np.max(ydata2)
        self.__plot9_ydelta = (self.__plot9_ymax-self.__plot9_ymin)/20
        self.__plot9_ax.set_ylim(self.__plot9_ymin-self.__plot9_ydelta, self.__plot9_ymax+self.__plot9_ydelta)
        self.__plot9_ax.set_title('Modeled Sr concentration profile')
        self.__plot9_ax.set_ylabel('Sr (ppm)')
        self.__plot9_ax.set_xlabel(r'Distance ($\mu$m)')

    @property
    def plot10_ax(self):
        return self.__plot10_ax
    @plot10_ax.setter
    def plot10_ax(self, ax):
        self.__plot10_ax = ax
    @property
    def plot10(self):
        return self.__plot10
    def plot10_func(self):
        self.__plot10_ax.clear()
        self.__plot10_ax.set_xlabel('Time (s)')
        self.__plot10_ax.set_ylabel(r'Amplitude [$Sr_{Obs}$/$Sr_{Eq}$ (1/$\mu$m)]')
        self.__plot10_ax.set_title('Exponential fit of rate of amplitude decrease')
        self.__plot10_ax.set_xlim(self.__time[0], self.__time[-1])
        ydata = self.__Ratio_slope_regress['Amp_IFT_reg']
        self.__plot10_ax.set_ylim(0, np.nanmax(ydata))
        dim = len(self.__freq)
        self.__plot10 = [[] for i in range(dim)]
        self.__plot10_cmap = pl.cm.jet(self.__freq)
        for i in range(dim):
            self.__plot10[i] = self.__plot10_ax.plot(self.__time, ydata[:,i], color=self.__plot10_cmap[i], linestyle = '-')
            if not self.__freq_filt[i]:
                self.__plot10[i][0].set_linestyle('None')
        self.__plot10_lim = self.__plot10_ax.plot([self.time_select, self.time_select], self.__plot10_ax.get_ylim(), 'k-')
        self.__plot10_type = 'regression'
        self.__plot10_ax.ticklabel_format(axis='y', style='sci', scilimits = (-2,3))
        
    def plot10_change(self):
        if self.__plot10:     
            
            for i in range(len(self.__freq)):
                self.__plot10[i][0].remove()
                if self.__plot10_type == 'regression':
                    ydata = self.__Ratio_slope_regress['Amp_IFT_res']
                    self.__plot10[i] = self.__plot10_ax.plot(self.__time, ydata[:,i], color=self.__plot10_cmap[i], linestyle = '-')
                else:
                    ydata = self.__Ratio_slope_regress['Amp_IFT_reg']
                    self.__plot10[i] = self.__plot10_ax.plot(self.__time, ydata[:,i], color=self.__plot10_cmap[i], linestyle = '-')
                if not self.__freq_filt[i]:
                    self.__plot10[i][0].set_linestyle('None')
            if self.__plot10_type == 'regression':
                self.__plot10_type = 'residual'
                self.__plot10_ax.set_ylabel(r'Residual [$Sr_{Obs}$/$Sr_{Eq}$ (1/$\mu$m)]')
                self.__plot10_ax.set_title('Residual of exponential fit')
                self.__plot10_ax.set_ylim(np.nanmin(ydata), np.nanmax(ydata))
            else:
                self.__plot10_type = 'regression'
                self.__plot10_ax.set_ylabel(r'Amplitude [$Sr_{Obs}$/$Sr_{Eq}$ (1/$\mu$m)]')
                self.__plot10_ax.set_title('Exponential fit of rate of amplitude decrease')
                self.__plot10_ax.set_ylim(0, np.nanmax(ydata))
                
    def reset_plots(self, val):
        
        if val == 1:
            if self.__plot1_ax != []:
                self.__plot1_ax.clear()
                self.__plot1 = []
                self.__plot1_ax = []
            if self.__plot2_ax != []:
                self.__plot2_ax.clear()
                self.__plot2 = []
                self.__plot2_ax = []
                self.__plot2_ax_twin.clear()
                self.__plot2_ax_twin.remove()
                self.__plot2_ax_twin = []
            if self.__plot3_ax != []:
                self.__plot3_ax.clear()
                self.__plot3 = []
                self.__plot3_ax = []
            
        if val in [1, 2]:
            if self.__plot4_ax != []:
                self.__plot4_ax.clear()
                self.__plot4 = []
                self.__plot4_ax = []
            if self.__plot5_ax != []:
                self.__plot5_ax.clear()
                self.__plot5 = []
                self.__plot5_ax = []
            if self.__plot6_ax != []:
                self.__plot6_ax.clear()
                self.__plot6 = []
                self.__plot6_ax = []
                
        if val in [1, 2, 3]:
            if self.__plot7_ax != []:
                self.__plot7_ax.clear()
                self.__plot7 = []
                self.__plot7_ax = []
            if self.__plot8_ax != []:
                self.__plot8_ax.clear()
                self.__plot8 = []
                self.__plot8_ax = []
            if self.__plot9_ax != []:
                self.__plot9_ax.clear()
                self.__plot9 = []
                self.__plot9_ax = []
            if self.__plot10_ax != []:
                self.__plot10_ax.clear()
                self.__plot10 = []
                self.__plot10_ax = []
            if self.__plot11_ax != []:
                self.__plot11_ax.clear()
                self.__plot11 = []
                self.__plot11_ax = []
            if self.__plot12_ax != []:
                self.__plot12_ax.clear()
                self.__plot12 = []
                self.__plot12_ax = []
            if self.__plot6_2:
                self.__plot6_2[0].remove()
                self.__plot6_2 = []
            if self.__plot3_2:
                self.__plot3_2[0].remove()
                self.__plot3_2 = []
                self.__plot3_2_textbox.remove()
            self.time_val = 0
    
    def sum_IFT(self, val):
        return self.__sum_IFT(val)
    
###############################################################################
class Plot_Profile():
    
    def __init__(self):
        plt.close('all')
        
        self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 9.5))
        self.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.9, wspace=0.3, hspace=0.3)
        self.fig.canvas.mpl_connect('close_event', self.fig_on_close)
        for i in range(2):
            for j in range(3):
                self.ax[i,j].set_visible(False)
        
        self.butt1_ax = self.fig.add_axes([0.46, 0.475, 0.08, 0.05])
        self.button1 = Button(self.butt1_ax, 'Open Profile', hovercolor='0.975')
        self.button1.on_clicked(self.open_file)
            
        self.fig2, self.ax2 = [], []
        self.profile = []
        
    def fig_on_close(self, event):
        if self.fig2 != []:
            plt.close(self.fig2)
    
    def open_file(self, event):
        root = tk.Tk()
        root.withdraw()
        file_name = fd.askopenfilename()
        if file_name != '':
            if self.fig2 != []:
                plt.close(self.fig2)
            if self.profile != []:
                self.slider1_ax.remove()
                self.slider2_ax.remove()
                self.rad1_ax.remove()
                self.check1_ax.remove()
                self.check2_ax.remove()
                self.check3_ax.remove()
                self.butt2_ax.remove()
                self.slider3_ax.remove()
                self.slider4_ax.remove()
                self.slider5_ax.remove()
                self.rad2_ax.remove()
                self.butt3_ax.remove()
                if self.profile.plot4_cbar:
                    self.profile.plot4_cbar.remove()
                    self.profile.plot4_cbar_ax.remove()
                    self.profile.plot4_cbar = []
                self.profile.reset_plots(1)
            print ('Open: ' + file_name)
            self.profile = Profile(file_name)
            self.butt1_ax.set_position([0.01, 0.95, 0.1, 0.04])
            self.button1.label.set_text('Change Profile')
            for i in range(2):
                for j in range(3):
                    self.ax[i,j].set_visible(True)
            self.ax_twin = self.ax[0,1].twinx()
            
            self.profile.fig = self.fig
            self.profile.plot1_ax = self.ax[0,0]
            self.profile.plot1_func()
            self.profile.plot2_ax = self.ax[0,1]
            self.profile.plot2_ax_twin = self.ax_twin
            self.profile.plot2_func()
            self.profile.plot3_ax = self.ax[0,2]
            self.profile.plot3_func()
            
            self.slider1_ax = self.fig.add_axes([0.382, 0.9, 0.256, 0.02])
            self.slider1 = RangeSlider(self.slider1_ax, '', self.profile.distance[0], self.profile.distance[-1],\
                           valinit=(self.profile.distance[0], self.profile.distance[-1]), valstep=self.profile.distance, color='r')
            self.slider1.valtext.set_text(round(self.slider1.val[1]-self.slider1.val[0],2))
            self.slider1.on_changed(self.slider1_update)
            self.slider1_ax.set_visible(False)
            
            self.slider2_ax = self.fig.add_axes([0.04488, 0.47, 0.26112, 0.02])
            self.log_time = np.log(self.profile.time)
            self.log_time = np.insert(self.log_time, 0, self.log_time[0]-((self.log_time[-1]-self.log_time[0])/50))
            self.slider2 = Slider(self.slider2_ax, '+T', self.log_time[0], self.log_time[-1],\
                           initcolor='k', color='lightgrey', valinit=self.log_time[1], valstep=self.log_time)
            self.slider2.on_changed(self.slider2_update)
            self.slider2.valtext.set_visible(False)
            
            self.rad1_ax = self.fig.add_axes([0.12, 0.92, 0.09, 0.07])
            self.radio1 = RadioButtons(self.rad1_ax, ['$Sr_{Obs}$ (ppm)', '$Sr_{Obs}$/$Sr_{Eq}$'], activecolor='0.3')
            self.radio1.on_clicked(self.radio1_update)
            
            self.check1_ax = self.fig.add_axes([0.215, 0.92, 0.09, 0.07])
            self.check1 = CheckButtons(self.check1_ax, ['Derivative', 'Subsample'])
            self.check1.on_clicked(self.check1_update)
            
            self.check2_ax = self.fig.add_axes([0.382, 0.92, 0.04, 0.07])
            self.check2 = CheckButtons(self.check2_ax, ['An', '$Sr_{Obs}$', '$Sr_{Eq}$'], actives=[True, True, True])
            self.check2.on_clicked(self.check2_update)
            c = ['w', 'b', 'g']    
            [rec.set_facecolor(c[i]) for i, rec in enumerate(self.check2.rectangles)]
            
            self.butt2_ax = self.fig.add_axes([0.01, 0.01, 0.1, 0.04])
            self.button2 = Button(self.butt2_ax, 'Calculate FT', hovercolor='0.975')
            self.button2.on_clicked(self.button2_update)
            
            self.slider3_ax = self.fig.add_axes([0.3823, 0.03, 0.256, 0.02])
            self.slider3 = RangeSlider(self.slider3_ax, 'Freq. filt.', self.profile.freq[0],\
                            self.profile.freq[-1], valstep=self.profile.freq, \
                            valinit=(self.profile.freq[0], self.profile.freq[-1]), color='r')
            self.slider3_ax.set_visible(False)
            self.slider3.on_changed(self.slider3_update)
            
            self.slider4_ax = self.fig.add_axes([0.65, 0.099, 0.012, 0.3479])
            self.slider4 = Slider(self.slider4_ax, 'Mag. thres.', 0, 1, valinit=0,\
                            color='lightgrey', orientation='vertical', initcolor='None')
            self.slider4_ax.set_visible(False)
            self.slider4.on_changed(self.slider4_update)
            
            self.rad2_ax = self.fig.add_axes([0.38, 0.45, 0.07, 0.06])
            self.radio2 = RadioButtons(self.rad2_ax, ['Freq. filt.', 'Mag. thres.', 'Freq. select'], activecolor='0.3')
            self.rad2_ax.set_visible(False)
            self.radio2.on_clicked(self.radio2_update)
            
            self.butt3_ax = self.fig.add_axes([0.89, 0.01, 0.1, 0.04])
            self.button3 = Button(self.butt3_ax, 'IFT regression', hovercolor='0.975')
            self.button3.on_clicked(self.button3_update)
            self.butt3_ax.set_visible(False)
            
            self.check3_ax = self.fig.add_axes([0.13, 0.01, 0.08, 0.04])
            self.check3 = CheckButtons(self.check3_ax, ['Tukey window'], actives=[self.profile.windowed])
            self.check3.on_clicked(self.check3_update)
            
            self.slider5_ax = self.fig.add_axes([0.23, 0.02, 0.05, 0.02])
            self.slider5 = Slider(self.slider5_ax, r'$\alpha$', 0, 1, initcolor='None',\
                                  color='lightgrey', valinit=self.profile.tukey_factor,\
                                  valstep=np.linspace(0, 1, 21))
            self.slider5.on_changed(self.slider5_update)
            self.slider5_ax.set_visible(self.profile.windowed)
            
    def slider5_update(self, val):
        self.profile.tukey_factor = val
    
    def slider1_update(self, val):
        self.slider1.valtext.set_text(round(val[1]-val[0],2))
        self.profile.dist_sub_idx = [np.where(self.profile.distance == val[0])[0][0], np.where(self.profile.distance == val[1])[0][0]]
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
            
    def slider2_update(self, val):
        self.profile.time_select_idx = np.where(self.log_time == val)[0][0]
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
        
    def radio1_update(self, label):
        if self.check1.get_status()[0]:
            if label == '$Sr_{Obs}$ (ppm)':
                self.profile.selected_array = 'Sr_diff_slope'
            else:
                self.profile.selected_array = 'Ratio_diff_slope'
        else:
            if label == '$Sr_{Obs}$ (ppm)':
                self.profile.selected_array = 'Sr_diff'
            else:
                self.profile.selected_array = 'Ratio_diff'
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
            
    def check1_update(self, label):
        if label == 'Derivative':
            if self.check1.get_status()[0]:
                if self.radio1.value_selected == '$Sr_{Obs}$ (ppm)':
                    self.profile.selected_array = 'Sr_diff_slope'
                else:
                    self.profile.selected_array = 'Ratio_diff_slope'
            else:
                if self.radio1.value_selected == '$Sr_{Obs}$ (ppm)':
                    self.profile.selected_array = 'Sr_diff'
                else:
                    self.profile.selected_array = 'Ratio_diff'
        else:
            if self.check1.get_status()[1]:
                self.slider1_ax.set_visible(True)
                self.slider1.valtext.set_text(round(self.slider1.val[1]-self.slider1.val[0],2))
                self.profile.plot1_lims[0][0].set_linestyle('-')
                self.profile.plot1_lims[1][0].set_linestyle('-')
                self.profile.plot2_lims[0][0].set_linestyle('-')
                self.profile.plot2_lims[1][0].set_linestyle('-')
            else:
                self.slider1_ax.set_visible(False)
                self.profile.dist_sub_idx = [0, len(self.profile.distance)-1]
                self.slider1.set_val((self.slider1.valmin, self.slider1.valmax))
                self.profile.plot1_lims[0][0].set_linestyle('None')
                self.profile.plot1_lims[1][0].set_linestyle('None')
                self.profile.plot2_lims[0][0].set_linestyle('None')
                self.profile.plot2_lims[1][0].set_linestyle('None')
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
    
    def check2_update(self, label):
        status = self.check2.get_status()
        if label == 'An':
            if status[0]:
                c = 'k'
            else:
                c = 'None'
            for i in range(3):
                self.profile.plot2[0][i][0].set_color(c)
        if label == '$Sr_{Obs}$':
            if status[1]:
                c = 'b'
            else:
                c = 'None'
            for i in range(3):
                self.profile.plot2[1][i][0].set_color(c)
        if label == '$Sr_{Eq}$':
            if status[2]:
                c = 'g'
            else:
                c = 'None'
            for i in range(3):
                self.profile.plot2[2][i][0].set_color(c)
        self.fig.canvas.draw_idle()
    
    def check3_update(self, label):
        self.profile.windowed = not self.profile.windowed
        self.slider5_ax.set_visible(self.profile.windowed)
        self.fig.canvas.draw_idle()
        
    def button2_update(self, event):
        if self.fig2 != []:
            self.profile.reset_plots(3)
            plt.close(self.fig2)
            self.fig2 = []
            self.ax2 = []
        if self.profile.plot4 != []:
            self.profile.reset_plots(2)
        self.profile.Calculate_FTs()
        self.slider3.valmin = self.profile.freq[0]
        self.slider3.valmax = self.profile.freq[-1]
        self.slider3_ax.set_xlim(self.profile.freq[0], self.profile.freq[-1])
        setattr(self.slider3,'valstep',self.profile.freq)
        self.slider3.set_val((self.profile.freq[0], self.profile.freq[-1]))
        self.button2.label.set_text('Recalculate FT')
        self.rad2_ax.set_visible(True)
        self.slider3_ax.set_visible(True)
        self.butt3_ax.set_visible(True)
        self.profile.plot4_ax = self.ax[1,0]
        self.profile.plot4_func()
        self.profile.plot5_ax = self.ax[1,1]
        self.profile.plot5_func()
        self.profile.plot5_lims[0][0].set_linestyle('-')
        self.profile.plot5_lims[1][0].set_linestyle('-')
        self.profile.plot6_ax = self.ax[1,2]
        self.profile.plot6_func()
        self.radio2.set_active(0)
        if self.check1.get_status()[0]:
            if self.radio1.value_selected == '$Sr_{Obs}$ (ppm)':
                self.profile.selected_array = 'Sr_diff_slope'                
            else:
                self.profile.selected_array = 'Ratio_diff_slope'
        else:
            if self.radio1.value_selected == '$Sr_{Obs}$ (ppm)':
                self.profile.selected_array = 'Sr_diff'
            else:
                self.profile.selected_array = 'Ratio_diff'
        self.slider4.valmax = self.profile.plot5_ymax+self.profile.plot5_ydelta
        self.slider4_ax.set_ylim(0, self.profile.plot5_ymax+self.profile.plot5_ydelta)
        self.fig.canvas.draw_idle()
        self.profile.regress_IFTs('reset')
        self.profile.mark_not_regress()
        
    def slider3_update(self, val):
        self.profile.freq_sub_idx = [np.where(self.profile.freq == val[0])[0][0], np.where(self.profile.freq == val[1])[0][0]]
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
            
    def slider4_update(self, val):
        self.profile.mag_filt = val
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
            
    def radio2_update(self, label):
        if label == 'Freq. filt.':
            ls = ['-', '-', 'None']
            lb = [True, False]
            self.slider4.set_val(0)
        elif label == 'Mag. thres.':
            ls = ['None', 'None', '-']
            lb = [False, True]
            self.slider3.set_val((self.slider3.valmin, self.slider3.valmax))
        else:
            ls = ['None', 'None', 'None']
            lb = [False, False]
            self.slider4.set_val(0)
            self.slider3.set_val((self.slider3.valmin, self.slider3.valmax))
        self.profile.filter_type = label
        self.profile.plot5_lims[0][0].set_linestyle(ls[0])
        self.profile.plot5_lims[1][0].set_linestyle(ls[1])
        self.profile.plot5_lims[2][0].set_linestyle(ls[2])
        self.slider3_ax.set_visible(lb[0])
        self.slider4_ax.set_visible(lb[1])
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.fig2.canvas.draw_idle()
            
    def button3_update(self, event):
        if self.fig2 != []:
            self.profile.reset_plots(3)
            plt.close(self.fig2)
            self.fig2 = []
            self.ax2 = []
        self.fig2, self.ax2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 9.5))
        self.fig2.subplots_adjust(left=0.075, bottom=0.1, right=0.955, top=0.9, wspace=0.3, hspace=0.3)
        self.fig2.canvas.mpl_connect('close_event', self.fig2_on_close)
        self.profile.plot7_ax = self.ax2[0,0]
        self.profile.plot7_func()
        self.profile.regress_IFTs()
        self.profile.plot10_ax = self.ax2[1,0]
        self.profile.plot10_func()
        self.profile.plot6_reg()
        self.profile.plot8_ax = self.ax2[0,1]
        self.profile.plot8_func()
        self.profile.plot9_ax = self.ax2[1,1]
        self.profile.plot9_func()
        self.profile.plot3_2_func()
        self.regress = True
        self.profile.mark_not_regress()
        
        self.slider6_ax = self.fig.add_axes([0.04488, 0.49, 0.26112, 0.02])
        self.slider6 = Slider(self.slider6_ax, '-T', self.log_time[0], self.log_time[-1],\
                       initcolor='k', color='lightgrey', valinit=self.log_time[1], valstep=self.log_time)
        self.slider6.on_changed(self.slider6_update)
        self.slider6.valtext.set_visible(False)
        
        self.butt4_ax = self.fig2.add_axes([0.36, 0.853, 0.09, 0.04])
        self.button4 = Button(self.butt4_ax, 'Log scale', hovercolor='0.975')
        self.button4.on_clicked(self.button4_update)
        
        self.butt5_ax = self.fig2.add_axes([0.36, 0.4, 0.09, 0.04])
        self.button5 = Button(self.butt5_ax, 'Show residual', hovercolor='0.975')
        self.button5.on_clicked(self.button5_update)
        
        self.check4_ax = self.fig2.add_axes([0.83, 0.01, 0.15, 0.04])
        self.check4 = CheckButtons(self.check4_ax, ['Correct to initial'], actives=[self.profile.correct_reg_init])
        self.check4.on_clicked(self.check4_update)
        
        self.fig.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
            
    def check4_update(self, label):
        if label == 'Correct to initial':
            self.profile.correct_reg_init = not self.profile.correct_reg_init
            self.fig2.canvas.draw_idle()
            self.fig.canvas.draw_idle()
            
    def fig2_on_close(self, event):
        self.profile.reset_plots(3)
        self.profile.regress_IFTs('reset')
        self.profile.mark_not_regress()
        self.fig.canvas.draw_idle()
        if self.fig2:
            self.slider6_ax.remove()
        self.fig2 = []
        self.ax2 = []
        self.profile.time_val = -3600
        
    def slider6_update(self, val):
        idx = np.where(self.log_time == val)[0][0]
        if idx == 0:
            self.profile.time_val = 0
        else:
            self.profile.time_val = -self.profile.time[idx-1]
        self.fig2.canvas.draw_idle()
        self.fig.canvas.draw_idle()
    
    def button4_update(self, event):
        
        if self.button4.label.get_text() == 'Log scale':
            self.profile.plot7_ax.set_xscale('log')
            if self.regress:
                self.profile.plot10_ax.set_xscale('log')
            self.button4.label.set_text('Linear scale')
        else:
            self.profile.plot7_ax.set_xscale('linear')
            if self.regress:
                self.profile.plot10_ax.set_xscale('linear')
            self.button4.label.set_text('Log scale')
    
    def button5_update(self, event):
        if self.regress:
            self.profile.plot10_change()
            self.button5.label.set_text('Show regression')
        
    
###############################################################################    
if __name__ == '__main__':
    if 'program' in locals():
        del program
    program = Plot_Profile()
    # a = r'C:\Users\mumob\Documents\Python Scripts\Sr_diff\16-3_diffused.xlsx'
    # prof = Profile(a)
    # l = len(prof.distance)
    # prof.dist_sub_idx = [0, int(l*1)]
    # prof.Calculate_FTs()
    # prof.regress_IFTs()
    # ift_reg = prof.Ratio_slope_regress
    # eval_1 = prof.IFT_regress_eval(ift_reg, 0.0)
    
    # plt.plot(prof.time, ift_reg['Amp_IFT_reg'][:,6],'g-')
    # plt.plot(prof.time, prof.Ratio_slope_IFT['Amp_IFT'][:,6],'k-')
    # prof.IFT_Amp_regress(array_name='Ratio_diff_slope')
    # reg = prof.Ratio_slope_regress
    # idx_reg = prof.filt_amps()
    
    # y = prof.Ratio_slope_IFT['Amp_IFT']
    # x = prof.time
    # idx = 20
    # plt.plot(x, y[:,idx], 'k-')
    # idx2 = 160
    # om = 10**(math.floor(math.log(prof.time[-1],10)))
    # def exp_fit(x, a, b, c):
    #     om = 10**math.floor(math.log(prof.time[-1],10))
    #     y = a*np.exp(b*x/om+c)
    #     return y
    
    # fit = curve_fit(exp_fit, prof.time[:idx2], y[:idx2,idx], p0=[1e-3,-50,1e-1], maxfev=int(1e4))
    # y2 = fit[0][0]*np.exp(fit[0][1]*prof.time/om+fit[0][2])
    # plt.plot(x[:],y2[:],'g-')
    # plt.plot(x[idx2],y2[idx2],'g.')
    # x3 = np.linspace(-1e5,x[0],10)
    # y3 = fit[0][0]*np.exp(fit[0][1]*x3/om+fit[0][2])
    # plt.plot(x3,y3,'g:')
    
    # l1 = len(x)
    # slope = np.zeros(l1-1)
    # for i in range(l1-1):
    #     slope[i] = (y[i+1,idx]-y[i,idx])/(x[i+1]-x[i])
    
    # fig_slope = plt.figure()
    # ax_slope = plt.axes()
    # ax_slope.plot(x[:-1],slope,'k.')
    
    # s2 = np.zeros(l1-2)
    # for i in range(l1-2):
    #     s2[i] = (slope[i+1]-slope[i])/(x[i+1]-x[i])
        
    #     fig_s2 = plt.figure()
    #     ax_s2 = plt.axes()
    #     ax_s2.plot(x[:-2],s2,'k.')

    # a = r'C:\Users\mumob\Documents\Python Scripts\Sr_diff\16-3_diffused.xlsx'
    # prof = Profile(a)
    # l = len(prof.distance)
    # prof.dist_sub_idx = [0, l]
    # prof.Calculate_FTs()
    # time = prof.time
    # amps = prof.Ratio_slope_IFT['Amp_IFT']
    # def filt_amps(time, amps):
    #     l = len(amps[0,:])
    #     t = len(time)
    #     idxs = np.zeros(l,dtype=int)
    #     sl = np.zeros(l)
    #     for f in range(l):
    #         for i in range(t-3):
    #             slope1 = (amps[i+1,f]-amps[i,f])/(time[i+1]-time[i])
    #             slope2 = (amps[i+2,f]-amps[i+1,f])/(time[i+2]-time[i+1])
    #             slope3 = (amps[i+3,f]-amps[i+2,f])/(time[i+3]-time[i+2])
    #             ds1 = (slope2-slope1)/(time[i+1]-time[i])
    #             ds2 = (slope3-slope2)/(time[i+2]-time[i+1])
    #             # or ds2 > ds1
    #             if slope1 >= 0 or slope2 >= 0 or ((slope2-slope1 > abs(slope1*0.2)) and (slope3-slope2 > abs(slope2*0.2))):
    #                 break
    #         idxs[f] = i
    #         sl[f] = slope1
    #     return idxs, sl
    
    # idx, sl = filt_amps(time, amps)
    # plt.plot(time, amps, 'k-')
    # for i in range(len(idx)):
    #     plt.plot(time[idx[i]], amps[idx[i],i], 'r.')