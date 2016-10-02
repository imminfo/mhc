
# coding: utf-8

# In[29]:

import os
import glob
import pandas as pd
import numpy as np

#your path to folder with .csv files
path = "C:/Users/Lev/.babun/cygwin/home/Lev/mhcflurry/experiments/combined-test-data/mhc/for_merge"
allfiles = glob.glob(os.path.join(path,"*.csv"))

np_array_list = []
for file_ in allfiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    
    np_array_list.append(df.as_matrix())

comb_np_array = np.vstack(np_array_list)
big_frame = pd.DataFrame(comb_np_array, columns=df.columns)
#save big dataframe to the same folder
big_frame.to_csv(path+"/Merged.csv")


# In[ ]:




# In[ ]:




# In[8]:




# In[ ]:




# In[ ]:



