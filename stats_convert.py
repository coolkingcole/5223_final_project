import arcpy
from arcpy import env
import numpy as np
import pandas as pd
#import geopandas#uncomment for pandas

import matplotlib.pyplot as plt
from scipy import stats

arcpy.env.overwriteOutput = True

aprx = arcpy.mp.ArcGISProject("CURRENT")

map = aprx.activeMap
if map == None:
    arcpy.AddError("Error getting map with aprx.activeMap")
    exit()

user_layer = arcpy.GetParameterAsText(0)#trt00_shp.shp
multi_field = arcpy.GetParameter(1)

for field in multi_field:
    pass

layer = map.addDataFromPath(user_layer)
fields_ndarr = arcpy.da.TableToNumPyArray(user_layer, "*", skip_nulls=True)
#total = field[output_dissolve_SUM_POP_name].sum()

arcpy.AddMessage(type(fields_ndarr))#<class 'numpy.ndarray'>
arcpy.AddMessage(type(user_layer))
arcpy.AddMessage(type(layer))
arcpy.AddMessage(fields_ndarr.dtype.names)

#this is a good way to get a pandas dataframe
#gdf = geopandas.read_file(user_layer)
#arcpy.AddMessage(type(gdf))
#rho = np.corrcoef(fields_ndarr['Area'])

# Make the first plot 
plt.plot(fields_ndarr['Area'])#, fields_ndarr['TRACTID'])


# Show the figure. 
#plt.show()


df_describe = pd.DataFrame(fields_ndarr['Area'])
arcpy.AddMessage(df_describe.describe())
