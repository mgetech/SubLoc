# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:26:41 2020

@author: MGE
"""
from datetime import datetime
from IPython.display import Audio
import numpy as np


def execution_time(start):
    end = datetime.now()
    elapsed = end - start
    print(f"\nIt took {elapsed.seconds + (elapsed.microseconds/1000000)} seconds to compute.")



def beeep():
    # Play an audio beep. Any audio URL will do.
    f1 = 400
    f2 = 1
    rate = 4000
    L = 1
    times = np.linspace(0,L,rate*L)
    signal = np.sin(2*np.pi*f1*times) + np.sin(2*np.pi*f2*times)    
    return Audio(data=signal, rate=rate, autoplay=True)


