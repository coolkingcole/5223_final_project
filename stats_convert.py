import arcpy
from arcpy import env
import numpy as np
import pandas as pd
#import geopandas#uncomment for pandas

import matplotlib.pyplot as plt
#from scipy import stats

###these will go in utils.py later
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
    arcprint(df.describe())

def correlate2D_ARR(df):
    #arcpy.AddMessage(df_describe.describe())
    pass


arcpy.env.overwriteOutput = True
aprx = arcpy.mp.ArcGISProject("CURRENT")
map = aprx.activeMap
if map == None:
    arcpy.AddError("Error getting map with aprx.activeMap")
    exit()

#Params
user_layer = arcpy.GetParameter(0)
fieldsOfIntrest = arcpy.GetParameter(1)
arcprint(fieldsOfIntrest[0])
makePlotBool = arcpy.GetParameter(2)
#arcprint(user_layer)
#multi_field = arcpy.GetParameter(1)

"""
for field in multi_field:
    pass
"""

layer = map.addDataFromPath(user_layer)
#fields_ndarr = arcpy.da.TableToNumPyArray(layer, "*", skip_nulls=True)
fields_ndarr = arcpy.da.FeatureClassToNumPyArray(layer, ('*'))#<class 'numpy.ndarray'>

if len(fieldsOfIntrest) == 1:
    #arcprint(fields_ndarr.dtype.names)
    #this is a good way to get a pandas dataframe
    #gdf = geopandas.read_file(user_layer)
    #arcpy.AddMessageQ(type(gdf))
    #rho = np.corrcoef(fields_ndarr['Area'])
    df = pd.DataFrame(fields_ndarr[str(fieldsOfIntrest[0])])
    #arcprint(df)
    #arcprint(df.describe())
    describe1D_ARR(df)
    
    if makePlotBool:
        makePlot(fields_ndarr,str(fieldsOfIntrest[0]))

elif len(fieldsOfIntrest) == 2:
    arcprint("2 fields hit")

elif len(fieldsOfIntrest) > 2:
    arcprint("greater than 2 fields hit")
