

#https://sdbs.db.aist.go.jp/sdbs/cgi-bin/direct_frame_top.cgi
#Carbon number 3-6.
#Hydrogen number 4-14.
#Other element all 0.
#Only with 1H NMR.
#Molecular weight 12-114.
#Molecular formular C%H%, exact match.
#Ascending order according to # of carbon.
#Chemical shift < 0 means no Hydrogen on that carbon.
#Index of carbon followed by type of bond.
#For example, if carbon 1 connects with carbon 2 via single bond and 4 with triple bond, then its bonding is 2,1,4,3.
#If the Hydrogens on one carbon have different chemical shift, use the average.

import numpy as np
import re

base = []
while True:
    try:
        SBS = int(input('SBS No.: '))
        if SBS < 0:
            break
        length = int(input('Length: '))
        A = np.zeros([length,length])
        for i in range(length):
            string = str(input('Bond #' + str(i + 1) + ' carbon connects to: '))
            info = list(map(int, re.split(',',string)))
            for j in range(int(len(info)/2)):
                A[i,info[2*j] - 1] = info[2*j + 1]
        string = str(input(['Chemical shift: ']))
        if len(string) > 0:
            shift = list(map(float, re.split(',', string)))
            while len(shift) != length:
                shift = list(map(float, re.split(',', str(input(['Not enough number, enter again: '])))))
        solvent = str(input('Solvent: '))
        print([SBS, A, shift, solvent])
        string = str(input('Want to correct? Y/N '))
        if len(string) > 0 and string[0] in ['y', 'Y']:
            continue
        base.append([SBS, A, shift, solvent])
    except:
        continue

filename = str(base[0][0])+'_'+str(base[-1][0])
np.save(filename, base, allow_pickle = True)
