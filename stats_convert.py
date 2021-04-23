import arcpy
from arcpy import env
import numpy as np
import pandas as pd
#import geopandas#uncomment for pandas
import matplotlib.pyplot as plt

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
user_layer2 = arcpy.GetParameter(1)
user_layer_text = arcpy.GetParameterAsText(0)#used to check if none
user_layer2_text = arcpy.GetParameterAsText(1)#used to check if none

fieldsOfInterest = arcpy.GetParameter(2)
fieldsOfInterest2 = arcpy.GetParameter(3)
#arcprint(fieldsOfInterest[0])
makePlotBool = arcpy.GetParameter(4)
try:
    layer = map.addDataFromPath(user_layer)
except:
    layer = user_layer

#fields_ndarr = arcpy.da.TableToNumPyArray(layer, "*", skip_nulls=True)
fields_ndarr = arcpy.da.FeatureClassToNumPyArray(layer, ('*'))#<class 'numpy.ndarray'>
if user_layer2_text != "":
    try:
        layer2 = map.addDataFromPath(user_layer2)
    except:
        layer2 = user_layer2
    fields_ndarr2 = arcpy.da.FeatureClassToNumPyArray(layer2, ('*'))#<class 'numpy.ndarray'>
    #fields_ndarr = np.concatenate([fields_ndarr,fields_ndarr2])
    #fields_ndarr=np.hstack([fields_ndarr, fields_ndarr2])


if len(fieldsOfInterest+fieldsOfInterest2) == 1:
    arcprint("1 field hit")
    #rho = np.corrcoef(fields_ndarr[str(fieldsOfInterest[0])])
    
    ####this works
    #df = pd.DataFrame(fields_ndarr[str(fieldsOfInterest[0])])
    #describe1D_ARR(df)
    df2 = arcgis_table_to_df(layer)
    describe1D_ARR(df2[str(fieldsOfInterest[0])])
    
    if makePlotBool:
        makePlot(fields_ndarr,str(fieldsOfInterest[0]))

elif len(fieldsOfInterest+fieldsOfInterest2) == 2:
    arcprint("2 fields hit")
    
    fields = []
    for f in fieldsOfInterest:
        fields.append(f)
    for f in fieldsOfInterest2:
        fields.append(f)
    
    if user_layer2_text == "":
        df = pd.DataFrame(fields_ndarr, columns = [str(fields[0]),str(fields[1])])
        column_1 = df[str(fields[0])]
        column_2 = df[str(fields[1])]
    else:
        df = pd.DataFrame(fields_ndarr, columns = [str(fields[0])])
        df2 = pd.DataFrame(fields_ndarr2, columns = [str(fields[1])])
        column_1 = df[str(fields[0])]
        column_2 = df2[str(fields[1])]

    pearson_correlation = column_1.corr(column_2,method='pearson')
    arcprint(pearson_correlation)
    pearson_correlation = column_1.corr(column_2,method='kendall')
    arcprint(pearson_correlation)
    pearson_correlation = column_1.corr(column_2,method='spearman')
    arcprint(pearson_correlation)
    

elif len(fieldsOfInterest+fieldsOfInterest2) > 2:
    arcprint("greater than 2 fields hit")
    fields1 = []
    fields2 = []
    for f in fieldsOfInterest:
        fields1.append(str(f))
    for f in fieldsOfInterest2:
        fields2.append(str(f))
    if user_layer2_text == "":
        #df = arcgis_table_to_df(layer)
        df = pd.DataFrame(fields_ndarr, columns = fields1)
        # plotting correlation heatmap
        dataplot=sb.heatmap(df.corr())
        # displaying heatmap
        plt.show()
    else:
        #df = arcgis_table_to_df(layer)
        #df2 = arcgis_table_to_df(layer2)
        df = pd.DataFrame(fields_ndarr, columns = fields1)
        df2 = pd.DataFrame(fields_ndarr2, columns = fields2)
        bigdf = df.append(df2, ignore_index=True)
        # plotting correlation heatmap
        dataplot=sb.heatmap(bigdf.corr())
        # displaying heatmap
        plt.show()
