import arcpy
from arcpy import env
import numpy as np
import pandas as pd
#import geopandas#uncomment for pandas

import matplotlib.pyplot as plt
#from scipy import stats

###these will go in util later
def arcprint(anything):
    arcpy.AddMessage(type(anything))
    arcpy.AddMessage(anything)

def makePlot(ndarr,field):
    plt.plot(fields_ndarr[field])
    plt.show()

def ndarr2Pd(ndarr,field):
    df = pd.DataFrame(ndarr[field])
    return df

def describe1D_ARR(df):
    arcprint(df_describe.describe())

def correlate2D_ARR(df):
    #arcpy.AddMessage(df_describe.describe())
    pass




arcpy.env.overwriteOutput = True

aprx = arcpy.mp.ArcGISProject("CURRENT")

map = aprx.activeMap
if map == None:
    arcpy.AddError("Error getting map with aprx.activeMap")
    exit()

user_layer = arcpy.GetParameter(0)#trt00_shp.shp
#arcprint(user_layer)
#multi_field = arcpy.GetParameter(1)

"""
for field in multi_field:
    pass
"""

layer = map.addDataFromPath(user_layer)
fields_ndarr = arcpy.da.TableToNumPyArray(layer, "*", skip_nulls=True)
fields_ndarr = arcpy.da.FeatureClassToNumPyArray(layer, ('*'))
arcprint(fields_ndarr)
#total = field[output_dissolve_SUM_POP_name].sum()
"""
arcprint(fields_ndarr)#<class 'numpy.ndarray'>
arcprint(user_layer)
arcprint(layer)
arcprint(fields_ndarr.dtype.names)
"""
#this is a good way to get a pandas dataframe
#gdf = geopandas.read_file(user_layer)
#arcpy.AddMessageQ(type(gdf))
#rho = np.corrcoef(fields_ndarr['Area'])

# Make the first plot 
#, fields_ndarr['TRACTID'])


# Show the figure. 

#makePlot(fields_ndarr,'EnrollTota')

#df_describe = ndarr2Pd(fields_ndarr,'AApct')
#describe1D_ARR(df_describe)
df = pd.DataFrame(fields_ndarr['ASpct'])
arcprint(df)
#arcprint(df.describe())

makePlot(fields_ndarr,'EnrollTota')
