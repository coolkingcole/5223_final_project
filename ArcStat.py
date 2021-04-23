import arcpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


###these will go in utils.py later
def arcprint(anything):
    arcpy.AddMessage(type(anything))
    arcpy.AddMessage(anything)


def makePlot(ndarr, field, msg):
    title = 'Plot of ' + field
    plt.xlabel(msg)
    plt.ylabel(field)
    plt.title(title)
    plt.plot(ndarr[field])
    plt.show()


def ndarr2Pd(ndarr, field):
    df = pd.DataFrame(ndarr[field])
    return df


def describe1D_ARR(df):
    arcprint(df.describe())


def correlate2D_ARR(df):
    # arcpy.AddMessage(df_describe.describe())
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
    data = [row for row in arcpy.da.SearchCursor(in_fc, final_fields, where_clause=query)]
    fc_dataframe = pd.DataFrame(data, columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName, drop=True)
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

# Params
user_layer = arcpy.GetParameter(0)
fieldsOfInterest = arcpy.GetParameter(1)
user_layer1 = arcpy.GetParameter(2)
user_layer1_text = arcpy.GetParameterAsText(2)
fieldsOfInterest1 = arcpy.GetParameter(3)
makePlotBool = arcpy.GetParameter(4)

try:
    layer = map.addDataFromPath(user_layer)
except:
    layer = user_layer

fields_ndarr = arcpy.da.FeatureClassToNumPyArray(layer, ('*'))  # <class 'numpy.ndarray'>
fields_ndarr1 = []
if user_layer1_text != "":
    try:
        layer1 = map.addDataFromPath(user_layer1)
    except:
        layer1 = user_layer1
    fields_ndarr1 = arcpy.da.FeatureClassToNumPyArray(layer1, ('*'))  # <class 'numpy.ndarray'>

if len(fields_ndarr1) > 0:
    minlen = min(len(fields_ndarr), len(fields_ndarr1))
    if len(fields_ndarr) > minlen:
        fields_ndarr = fields_ndarr[:minlen]
    if len(fields_ndarr1) > minlen:
        fields_ndarr1 = fields_ndarr1[:minlen]
fields = []
for f in fieldsOfInterest:
    fields.append(str(f))
fields1 = []
if len(fields_ndarr1) > 0:
    for f in fieldsOfInterest1:
        fields1.append(str(f))

df = pd.DataFrame(fields_ndarr, columns=fields)

df2 = df
if len(fields_ndarr1) > 0:
    df1 = pd.DataFrame(fields_ndarr1, columns=fields1)
    df2 = df_append(df, df1)
fields2 = df2.columns.tolist()  # calculate fields here to drop duplicated fields

if len(fields2) == 1:
    column1 = df2[fields2[0]]
    max1 = column1.max()
    min1 = column1.min()
    mean1 = column1.mean()
    median1 = column1.median()
    std1 = column1.std()
    linemsg = 'max:{:.2f},min:{:.2f},mean:{:.4f},median:{:.4f},std:{:.4f}'.format(max1, min1, mean1, median1, std1)
    arcpy.AddMessage('For ' + fields2[0] + ',' + linemsg)
    if makePlotBool:
        makePlot(df2, fields2[0], linemsg)

elif len(fields2) == 2:
    column_1 = df2[fields2[0]]
    column_2 = df2[fields2[1]]
    pearson_correlation = column_1.corr(column_2, method='pearson')
    corr_title = 'Pearson\'s Correlation = {:.4f}'.format(pearson_correlation)
    arcpy.AddMessage(
        "Pearson correlation coefficient between {} and {} is:{}\n".format(fields2[0],
                                                                           fields2[1], pearson_correlation))
    pearson_correlation = column_1.corr(column_2, method='kendall')
    arcpy.AddMessage(
        "Kendall correlation coefficient between {} and {} is:{}\n".format(fields2[0], fields2[1], pearson_correlation))

    pearson_correlation = column_1.corr(column_2, method='spearman')
    arcpy.AddMessage(
        "Spearman correlation coefficient between {} and {} is:{}\n".format(fields2[0],
                                                                            fields2[1], pearson_correlation))
    if makePlotBool:
        arcpy.AddMessage("Draw plot of {} versus {}\n".format(fields2[1], fields2[0]))
        makePlot2(df2, fields2[0], fields2[1], corr_title)


elif len(fields2) > 2:
    # todo make sure only wanted fields are shown
    corr = corranalysis(df2, 'pearson')
    arcpy.AddMessage(
        "Pearson correlation coefficient for fields {} is:\n{}\n".format(fields2, corr))
    corr = corranalysis(df2, 'kendall')
    arcpy.AddMessage(
        "Kendall correlation coefficient for fields {} is:\n{}\n".format(fields2, corr))
    corr = corranalysis(df2, 'spearman')
    arcpy.AddMessage(
        "Spearman correlation coefficient for fields {} is:\n{}\n".format(fields2, corr))

    if makePlotBool:
        try:
            import seaborn as sb

            arcpy.AddMessage("Draw heatmap for fields {}\n".format(fields2))
            # plotting correlation heatmap
            dataplot = sb.heatmap(df2.corr(), annot=True, fmt=".4f")
            # displaying heatmap
            plt.title('Heatmap of Pearson correlation coefficient')
            plt.show()
        except Exception:
            arcpy.AddWarning("Package seaborn is not installed, heatmap can not be generated!")
