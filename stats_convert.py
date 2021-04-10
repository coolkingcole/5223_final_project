import arcpy
from arcpy import env
import numpy as np
import pandas as pd
#import geopandas#uncomment for pandas

import matplotlib.pyplot as plt
#from scipy import stats
from sklearn.linear_model import LinearRegression


import seaborn as sb
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

def arcgis_table_to_df(in_fc, input_fields=None, query=""):
    """Function will convert an arcgis table into a pandas dataframe with an object ID index, and the selected
    input fields using an arcpy.da.SearchCursor.
    :param - in_fc - input feature class or table to convert
    :param - input_fields - fields to input to a da search cursor for retrieval
    :param - query - sql query to grab appropriate values
    :returns - pandas.DataFrame"""
    OIDFieldName = arcpy.Describe(in_fc).OIDFieldName
    if input_fields:
        final_fields = [OIDFieldName] + input_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_fc)]
    data = [row for row in arcpy.da.SearchCursor(in_fc,final_fields,where_clause=query)]
    fc_dataframe = pd.DataFrame(data,columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName,drop=True)
    return fc_dataframe

def arcgis_table_to_dataframe(in_fc, input_fields, query="", skip_nulls=False, null_values=None):
    """Function will convert an arcgis table into a pandas dataframe with an object ID index, and the selected
    input fields. Uses TableToNumPyArray to get initial data.
    :param - in_fc - input feature class or table to convert
    :param - input_fields - fields to input into a da numpy converter function
    :param - query - sql like query to filter out records returned
    :param - skip_nulls - skip rows with null values
    :param - null_values - values to replace null values with.
    :returns - pandas dataframe"""
    OIDFieldName = arcpy.Describe(in_fc).OIDFieldName
    if input_fields:
        final_fields = [OIDFieldName] + input_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_fc)]
    np_array = arcpy.da.TableToNumPyArray(in_fc, final_fields, query, skip_nulls, null_values)
    object_id_index = np_array[OIDFieldName]
    fc_dataframe = pd.DataFrame(np_array, index=object_id_index, columns=input_fields)
    return fc_dataframe

arcpy.env.overwriteOutput = True
aprx = arcpy.mp.ArcGISProject("CURRENT")
map = aprx.activeMap
if map is None:
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
    arcprint("1 field hit")
    #rho = np.corrcoef(fields_ndarr[str(fieldsOfIntrest[0])])
    
    ####this works
    #df = pd.DataFrame(fields_ndarr[str(fieldsOfIntrest[0])])
    #describe1D_ARR(df)
    ####
    #######This one is nicer
    df2 = arcgis_table_to_df(layer)
    describe1D_ARR(df2[str(fieldsOfIntrest[0])])
    
    if makePlotBool:
        makePlot(fields_ndarr,str(fieldsOfIntrest[0]))

elif len(fieldsOfIntrest) == 2:
    arcprint("2 fields hit")
    
    fields = []
    for f in fieldsOfIntrest:
        fields.append(f)

    df = pd.DataFrame(fields_ndarr, columns = [str(fields[0]),str(fields[1])])
    #df = pd.DataFrame(fields_ndarr, columns=fields_ndarr.class_names)
    column_1 = df[str(fields[0])]
    column_2 = df[str(fields[1])]
    arcprint(column_1)
    pearson_correlation = column_1.corr(column_2,method='pearson')
    arcprint(pearson_correlation)
    pearson_correlation = column_1.corr(column_2,method='kendall')
    arcprint(pearson_correlation)
    pearson_correlation = column_1.corr(column_2,method='spearman')
    arcprint(pearson_correlation)
    

elif len(fieldsOfIntrest) > 2:
    arcprint("greater than 2 fields hit")
    fields = []
    for f in fieldsOfIntrest:
        fields.append(f)
    
    df2 = arcgis_table_to_df(layer)
    # plotting correlation heatmap
    dataplot=sb.heatmap(df2.corr())
    # displaying heatmap
    plt.show()
