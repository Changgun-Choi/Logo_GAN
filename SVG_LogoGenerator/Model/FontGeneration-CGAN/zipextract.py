#!/usr/bin/env python
# coding: utf-8

# In[6]:


import zipfile
with zipfile.ZipFile('A_Z_Images.zip', 'r') as zip_ref:
    zip_ref.extractall('A_Z_Images')

