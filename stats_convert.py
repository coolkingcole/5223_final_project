import arcpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sb
###these will go in utils.py later
def arcprint(anything):
    arcpy.AddMessage(type(anything))
    arcpy.AddMessage(anything)

def makePlot(ndarr,field):
    plt.plot(fields_ndarr0[field])
    plt.show()

def ndarr2Pd(ndarr,field):
    df = pd.DataFrame(ndarr[field])
    return df

def describe1D_ARR(df):
    arcprint(df.describe())

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

def df_append(df1, df2):
    """Function will append two pandas dataframe.
    :param - df1 - Input pandas dataframe
    :param - df2 - Input pandas dataframe
    :returns - A new pandas dataframe with all columns of two input pandas dataframe"""
    result = copy.deepcopy(df1)
    for col_name in df2.columns.tolist():
        result[col_name] = df2[col_name]
    return result


def makePlot2(dfparm, field1, field2, title):
    plt.xlabel(field1)
    plt.ylabel(field2)
    plt.title(title)
    plt.plot(dfparm[field1], dfparm[field2], 'bo')
    plt.show()


def corranalysis(df, method):
    """Function will apply specified correlation analysis to panadas dataframe.
    :param - df - Input pandas dataframe
    :param - method - Input method like 'pearson', 'kendall', 'spearman'
    :returns - A pandas dataframe with analysis result
    """
    df_corr = df.corr(method=method, min_periods=1)
    return df_corr

arcpy.env.overwriteOutput = True
aprx = arcpy.mp.ArcGISProject("CURRENT")
map = aprx.activeMap
if map is None:
    arcpy.AddError("Error getting map with aprx.activeMap")
    exit()

#Params
user_layer0 = arcpy.GetParameter(0)
user_layer1 = arcpy.GetParameter(1)
user_layer0_text = arcpy.GetParameterAsText(0)#used to check if none
user_layer1_text = arcpy.GetParameterAsText(1)#used to check if none

fieldsOfInterest = arcpy.GetParameter(2)
fieldsOfInterest2 = arcpy.GetParameter(3)

makePlotBool = arcpy.GetParameter(4)
try:
    layer0 = map.addDataFromPath(user_layer0)
except:
    layer0 = user_layer0

fields_ndarr0 = arcpy.da.FeatureClassToNumPyArray(layer0, ('*'))#<class 'numpy.ndarray'>
if user_layer1_text != "":
    try:
        layer1 = map.addDataFromPath(user_layer1)
    except:
        layer1 = user_layer1
    fields_ndarr1 = arcpy.da.FeatureClassToNumPyArray(layer1, ('*'))#<class 'numpy.ndarray'>



if len(fieldsOfInterest+fieldsOfInterest2) == 1:
    arcprint("1 field hit")
    ####this works
    #df = pd.DataFrame(fields_ndarr0[str(fieldsOfInterest[0])])
    #describe1D_ARR(df)
    df2 = arcgis_table_to_df(layer0)
    describe1D_ARR(df2[str(fieldsOfInterest[0])])
    
    if makePlotBool:
        makePlot(fields_ndarr0,str(fieldsOfInterest[0]))

elif len(fieldsOfInterest+fieldsOfInterest2) == 2:
    arcprint("2 fields hit")
    
    fields = []
    for f in fieldsOfInterest:
        fields.append(f)
    for f in fieldsOfInterest2:
        fields.append(f)
    
    if user_layer1_text == "":
        df = pd.DataFrame(fields_ndarr0, columns = [str(fields[0]),str(fields[1])])
        column_1 = df[str(fields[0])]
        column_2 = df[str(fields[1])]
    else:
        df = pd.DataFrame(fields_ndarr0, columns = [str(fields[0])])
        df2 = pd.DataFrame(fields_ndarr1, columns = [str(fields[1])])
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
    if user_layer1_text == "":
        #df = arcgis_table_to_df(layer)
        df = pd.DataFrame(fields_ndarr0, columns = fields1)
        # plotting correlation heatmap
        dataplot=sb.heatmap(df.corr())
        # displaying heatmap
        plt.show()
    else:
        #df = arcgis_table_to_df(layer)
        #df2 = arcgis_table_to_df(layer1)
        df = pd.DataFrame(fields_ndarr0, columns = fields1)
        df2 = pd.DataFrame(fields_ndarr1, columns = fields2)
        bigdf = df.append(df2, ignore_index=True)
        # plotting correlation heatmap
        dataplot=sb.heatmap(bigdf.corr())
        # displaying heatmap
        plt.show()
