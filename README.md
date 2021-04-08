# 5223_final_project

Final project for 5223

# Idea

Make a arcgis script that can calculate different statistics and analytics from a user supplied file.

# Libraries
* arcpy
* matplotlib
* numpy
* pandas
* seaborn
* scikit-learn

# Release 1

## Basic interface and function

Descriptive stats  
Correlation for multiple feilds  
Basic user interface  
User Manual

# Fields
#### (so far)
![image](./misc_assets/final_params_v0.PNG)

# User Manual
#todo
So far Arcstat will do different analysis depending on how many columns are selected. If one column is selected, descriptive statistics will be printed. If 2 fields are given correlation stats will be printed if more are given...

# Vision Statemenmt
***For*** ArgGIS Pro users

***Who*** need to analyze data and find many different descriptive statistics and correlations between multiple data sources.

***The*** ArcStat plugin is a tool

***That*** can present descriptive statistics and correlation of multiple fields in the form of chart to give the user as much information about the data they are attempting to analyze and give useful output to power further analysis in ArcGIS or other tools

***Unlike*** individual analysis tools in the ArcGIS built-in toolboxes

***Our*** productprovides all these statistical analysis in one place in an intuitive way that can output information to the user or to formats that can be used in other tools.

# Installation  
Run ```python -m pip install -r requirements```  
This should solve any "ModuleNotFoundError: No module named X" errors  

Installation instructions are under the "Using geoprocessing packages" heading in this material from OSU GEOG 5223
[link to docs](/misc_assets/script-tools.html)
