import DataScience.MachineLearning as ai
FeatureSelection = ai.Preprocessing()
df1 = FeatureSelection.LoadData("Sample2.csv")
df2 = FeatureSelection.LoadData("Sample.tsv",HeaderMissing="Yes")
df3 = FeatureSelection.LoadData("Sample.xls")
df4 = FeatureSelection.LoadData("Sample.xlsx")
df5 = FeatureSelection.LoadData("Sample2.json")
df6 = FeatureSelection.LoadData("Sample1.xml")
df7 = FeatureSelection.LoadData("SampleORC.orc")
df8 = FeatureSelection.LoadData("SampleParquet.parquet")
df9 = FeatureSelection.LoadData("SampleAvro.avro")

import json 
import pandas as pd 
from flatten_json import flatten
from pandas.io.json import json_normalize
FileName = 'Sample5.json'
with open(FileName) as RequiredFile:
    json = json.load(RequiredFile)
if isinstance(json, dict): 
    if(len(json) > 1):
        DataFrame = json_normalize(flatten(json))
    else:
        DataFrame = json_normalize(list(json.values())[0]) 
else:
    FlattenedData = (flatten(_json) for _json in json)
    DataFrame = pd.DataFrame(FlattenedData)


import pandas as pd
import pyarrow.orc as orc
import pyarrow.parquet as parquet

with open("SampleORC", encoding="utf-16", errors='ignore') as file:
#with open("SampleORC") as file:
    data = orc.ORCFile(file)
    df = data.read().to_pandas()

data = orc.ORCFile("SampleORC.orc")
df = data.read().to_pandas()

data2 = parquet.ParquetFile("SampleParquet.parquet")
df2 = data2.read().to_pandas()

from fastavro import reader
with open('SampleAvro.avro', 'rb') as fo:
    for record in reader(fo):
        print(record)


import numpy as np
import pandas as pd
import pandavro as pdx

df = pd.DataFrame({"Boolean": [True, False, True, False],
                   "Float64": np.random.randn(4),
                   "Int64": np.random.randint(0, 10, 4),
                   "String": ['foo', 'bar', 'foo', 'bar'],
                   "DateTime64": [pd.Timestamp('20190101'), pd.Timestamp('20190102'),
                                  pd.Timestamp('20190103'), pd.Timestamp('20190104')]})

pdx.to_avro("SampleAvro2.avro", df)
saved = pdx.read_avro("SampleAvro.avro")
print(saved)


vv = pd.read_csv("Sample2.csv", header=0)
bb = pd.read_csv("Sample.tsv", sep='\t', header=None)
aa= pd.read_excel('Sample.xls')
aa2= pd.read_excel('Sample.xlsx')
aa.columns = ['a', 'b', 'c', 'd', 'e', 'f']


'''
FileName = '/home/contactrkk_gmail/1D311A1E02824594/AllKindOfStuff/ML/Salaries.csv'
FileType = FileName.split(".")
FileType = FileType[len(FileType)-1].lower()
observations = pd.read_csv('Salaries.csv')
'''

from urllib.request import urlopen
html = urlopen("http://www.google.com/")
print(html)

from urllib2 import Request, urlopen
import json
from pandas.io.json import json_normalize

path1 = '42.974049,-81.205203|42.974298,-81.195755'
request=Request('http://maps.googleapis.com/maps/api/elevation/json?locations='+path1+'&sensor=false')
response = urlopen(request)
elevations = response.read()
data = json.loads(elevations)
json_normalize(data['results'])

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

import json 
import pandas as pd 
from pandas.io.json import json_normalize #package for flattening json in pandas df

#load json object
with open('Sample6.json') as f:
    d = json.load(f)

xx=json_normalize(d)

from flatten_json import flatten
dic_flattened = (flatten(xd) for xd in d)
import pandas as pd
df = pd.DataFrame(dic_flattened)

nycphil = json_normalize(list(d.values())[0])
nycphil.head(3)

flat = flatten_json(d)
xx2 = json_normalize(flat)

import pandas as pd
import xml.etree.ElementTree as et

FileName = '/home/contactrkk_gmail/Minus1By12DataScience/DataScience/SampleData/Books.xml'
#FileName = '/home/contactrkk_gmail/Minus1By12DataScience/DataScience/SampleData/Employees.xml'

RootElement = et.parse(FileName).getroot()
RootElementTag = RootElement.tag
RootElementAttributes = []

for Item in RootElement.keys():
    if "__"+RootElementTag+"___"+Item not in RootElementAttributes :
        RootElementAttributes.append("__"+RootElementTag+"___"+Item)
        
CoreElement = []
CoreElementAttributes = []
CoreNodes = []
CoreNodesAttributes = []
FinalColumns = []

for CE in RootElement: 
    if CE.tag not in CoreElement :
        CoreElement.append(CE.tag) 
    for Item in CE.keys():
        if CE.tag+"___"+Item not in CoreElementAttributes :
            CoreElementAttributes.append(CE.tag+"___"+Item)
    for Item in list(CE):
        if CE.tag+"__"+Item.tag not in CoreNodes :
            CoreNodes.append(CE.tag+"__"+Item.tag)   
        for Item_ in Item.keys():
            if CE.tag+"__"+Item.tag+"___"+Item_ not in CoreNodesAttributes :
                CoreNodesAttributes.append(CE.tag+"__"+Item.tag+"___"+Item_)

RootElementAttributes = sorted(RootElementAttributes) 
CoreElement = sorted(CoreElement) 
CoreElementAttributes = sorted(CoreElementAttributes) 
CoreNodes = sorted(CoreNodes) 
CoreNodesAttributes = sorted(CoreNodesAttributes) 
FinalColumns = FinalColumns+RootElementAttributes+CoreElementAttributes+CoreNodes+CoreNodesAttributes
FinalColumns = sorted(FinalColumns) 
DataFrame = pd.DataFrame(columns = FinalColumns)

for CE in RootElement: 
    DataRow = []
    for Item in RootElementAttributes:
        DataRow.append(RootElement.attrib.get(Item.split("___")[1]))
    for Item in CoreElementAttributes:
        DataRow.append(CE.attrib.get(Item.split("___")[1]))        
    for Item in CoreNodes:
        if CE is not None and CE.find(Item.split("__")[1]) is not None:
            DataRow.append(CE.find(Item.split("__")[1]).text)
        else: 
            DataRow.append(None)     
        CoreNodesAttributesFiltered = [Value for Value in CoreNodesAttributes if Value.split("___")[0] == Item]
        for CNAF in CoreNodesAttributesFiltered:
            DataRow.append(CE.find(Item.split("__")[1]).attrib.get(CNAF.split("___")[1]))
            #print(CE.find(Item.split("__")[1]).attrib)
            #print("**********")
        #print(CoreNodesAttributesFiltered)
        #print("----------------")
    #print(DataRow)   
    DataFrame = DataFrame.append(pd.Series(DataRow, index = FinalColumns), ignore_index = True)    
'''   
#df_cols = ["id", "author", "title", "genre", "price", "publish_date", "description"] 
    for item in list(node):
        print(item.tag)
        #print(item.attrib)
        print(item.keys())
        
        
    #print(node)
    print(node.tag)
    #print(node.attrib)
    print(node.keys())
    #print(list(node))
    for item in list(node):
        print(item.tag)
        #print(item.attrib)
        print(item.keys())

xtree = et.parse(FileName)
#print(xtree)
xroot = xtree.getroot()
#print(xroot)
print(xroot.tag)
#print(xroot.attrib)
print(xroot.keys())
#print(xroot.items())
#out_df = pd.DataFrame(columns = df_cols)

for node in xroot: 
    #print(node)
    print(node.tag)
    #print(node.attrib)
    print(node.keys())
    #print(list(node))
    for item in list(node):
        print(item.tag)
        #print(item.attrib)
        print(item.keys())
 
'''
       
'''        
for node in xroot: 
    #print(node)
    print(node.tag)
    #print(node.attrib)
    print(node.keys())
    #print(list(node))
    for item in list(node):
        print(item.tag)
        #print(item.attrib)
        print(item.keys())
    res = []
    res.append(node.attrib.get(df_cols[0]))
    for el in df_cols[1:]: 
        if node is not None and node.find(el) is not None:
            res.append(node.find(el).text)
        else: 
            res.append(None)
    out_df = out_df.append(pd.Series(res, index = df_cols), ignore_index = True)



def parse_XML(xml_file, df_cols): 
    """Parse the input XML file and store the result in a pandas DataFrame 
    with the given columns. The first element of df_cols is supposed to be 
    the identifier variable, which is an attribute of each node element in 
    the XML data; other features will be parsed from the text content of 
    each sub-element. """
    
    xtree = et.parse(xml_file)
    print(xtree)
    xroot = xtree.getroot()
    print(xroot)
    out_df = pd.DataFrame(columns = df_cols)
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        out_df = out_df.append(pd.Series(res, index = df_cols), ignore_index = True)
        
    return out_df

xx = parse_XML(FileName, ["id", "author", "title", "genre", "price", "publish_date", "description"])

'''


