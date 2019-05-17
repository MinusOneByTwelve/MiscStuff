class Preprocessing:

#Assumption 1 - Data Columns For Train & Test Will Be Same  
#Assumption 2 - Ordinal & Bit Switches Will Not Be Pushed In Nominal Function    
#Assumption 3 - Train Categorical Will Be SuperSet & Test Will Be SubSet, Else Model To Be ReCreated

   def LoadData(self, FileName, HeaderMissing="No"):
    # Supports excel,csv,tsv,xml,json,orc,parquet,avro
    import pandas as pd
    FileType = FileName.split(".")
    FileType = FileType[len(FileType)-1].lower()
    if FileType == 'xls':
        if HeaderMissing =="Yes":
            return pd.read_excel(FileName, header=None) 
        else:
            return pd.read_excel(FileName)
    if FileType == 'xlsx':
        if HeaderMissing =="Yes":
            return pd.read_excel(FileName, header=None) 
        else:
            return pd.read_excel(FileName)   
    if FileType == 'csv':
        if HeaderMissing =="Yes":
            return pd.read_csv(FileName, header=None) 
        else:
            return pd.read_csv(FileName)
    if FileType == 'tsv':
        if HeaderMissing =="Yes":
            return pd.read_csv(FileName, header=None, sep='\t') 
        else:
            return pd.read_csv(FileName, sep='\t')    
    if FileType == 'orc':
        import pyarrow.orc as orc
        return orc.ORCFile(FileName).read().to_pandas()
    if FileType == 'parquet':
        import pyarrow.parquet as parquet
        return parquet.ParquetFile(FileName).read().to_pandas() 
    if FileType == 'avro':
        import pandavro as pdx
        return pdx.read_avro(FileName)    
    if FileType == 'json':
        import json 
        from flatten_json import flatten
        from pandas.io.json import json_normalize
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
        return DataFrame  
    if FileType == 'xml':
        import xml.etree.ElementTree as et
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
        return DataFrame
    
   def HandleCategorical_(self, DataObject):       
    from sklearn.preprocessing import LabelEncoder    
    labelencoder = LabelEncoder()    
    DataObject = labelencoder.fit_transform(DataObject)    
    return DataObject
    
   def HandleCategorical(self, DataObject, Ordinal=None, Nominal=None, Inference=None):       
    from sklearn.preprocessing import LabelEncoder
    import numpy as np   
    labelencoder = LabelEncoder()
    _Inference = {}  
    if Ordinal is not None: 
        Ordinal.sort()       
        _Inference["Ordinal"] = Ordinal        
        for Column in Ordinal: 
            _Inference[Column] = np.sort(np.unique(DataObject[:,Column]))
            DataObject[:, Column] = labelencoder.fit_transform(DataObject[:, Column])            
    if Nominal is not None: 
        Nominal.sort()       
        _Inference["Nominal"] = Nominal       
        for Column in Nominal: 
            _Inference[Column] = np.sort(np.unique(DataObject[:,Column]))
            DataObject[:, Column] = labelencoder.fit_transform(DataObject[:, Column])             
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            CatNominalColsTemp = Nominal.copy()            
        for Column in range(0,len(Nominal)):
            ColumnNo=CatNominalColsTemp[Column]
            #print(ColumnNo)
            NoOfColumnsToAdd=len(_Inference[Nominal[Column]])-2
            for _Column in range(Column+1,len(Nominal)):
                CatNominalColsTemp[_Column] = CatNominalColsTemp[_Column]+NoOfColumnsToAdd                
            onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [ColumnNo])],remainder='passthrough')
            DataObject = np.array(onehotencoder.fit_transform(DataObject), dtype=np.float)
            DataObject = DataObject[:, 1:]                        
        _Inference["Nominal_"] = CatNominalColsTemp           
    if Inference is not None: 
        _Inference = Inference
        TotalColumns = _Inference["Ordinal"]+_Inference["Nominal"] 
        IsDataNew = len(TotalColumns)
        NewData = {}
        for Column in TotalColumns:
            TempColumnData = np.sort(np.unique(DataObject[:,Column]))
            if(set(TempColumnData).issubset(set(_Inference[Column]))): 
                IsDataNew = IsDataNew-1
            else:
                NewData[Column] = (list(set(TempColumnData) - set(_Inference[Column]))) 
        _Inference["IsDataNew"] = IsDataNew
        _Inference["NewData"] = NewData
        if(IsDataNew < 1):
            DataRepresentation = {}
            for Column in _Inference["Ordinal"]:
                if(len(np.unique(DataObject[:,Column])) == len(_Inference[Column])):
                     DataRepresentation[Column] = 1
                     DataObject[:, Column] = labelencoder.fit_transform(DataObject[:, Column])
                else:
                     DataRepresentation[Column] = 0
                     for Row in DataObject:
                         Row[Column] = np.where(_Inference[Column] == Row[Column])[0][0]                
            _Inference["Ordinal_"] = DataRepresentation 
            for Index in range(0,len(_Inference["Nominal"])):
                Column = _Inference["Nominal_"][Index]
                OriginalColumn = _Inference["Nominal"][Index] 
                TempData = DataObject[:,Column]
                #print(TempData)
                #print(type(TempData))
                #print('-------------')                
                #print(DataObject)
                #print(type(DataObject))
                #print('=============')
                DataObject = np.delete(DataObject, Column, 1)
                #print(DataObject)
                #print(type(DataObject))
                #print('/////////////')
                for Count in range(0,len(_Inference[OriginalColumn])):
                    DataObject = np.append(arr = np.zeros((len(DataObject),1)).astype(int), values = DataObject, axis = 1)  
                #print(DataObject)
                #print(type(DataObject))
                #print('*************')                
                for _Index in range(0,len(TempData)):
                    DataObject[_Index,(np.where(_Inference[OriginalColumn] == TempData[_Index])[0][0])] = 1  
                #print(DataObject)
                #print(type(DataObject))
                #print('~~~~~~~~~~~~~')
                DataObject = np.array(DataObject)
                #print(type(DataObject))                
                DataObject = DataObject[:, 1:] 
                #print(DataObject)
                #print(type(DataObject))
                #print('|||||||||||||')  
            DataObject = DataObject.astype(float)                                       
    return DataObject,_Inference  

   def ReduceMemoryFootprint(self, DataObject):     
    import numpy as np
    import pandas as pd
    start_mem_usg = DataObject.memory_usage().sum() / 1024**2 
    print("*******************************************")
    print("Current Usage : ",start_mem_usg," MB")
    NullDataColumns = []
    Inference = {'ColumnName': [],'OriginalDataType': [],'ModifiedDataType': [],'HasNullData': []}
    for col in DataObject.columns:
        Inference['ColumnName'].append(col)
        Inference['OriginalDataType'].append(DataObject[col].dtype)
        Inference['HasNullData'].append(DataObject[col].isnull().any())
        if DataObject[col].dtype != object:            
            #print("******************************")
            #print("Column : ",col)
            #print("Current DataType : ",DataObject[col].dtype)
            
            IsInt = False
            mx = DataObject[col].max()
            mn = DataObject[col].min()
            
            if not np.isfinite(DataObject[col]).all(): 
                NullDataColumns.append(col)
                DataObject[col].fillna(mn-1,inplace=True)  
                   
            asint = DataObject[col].fillna(0).astype(np.int64)
            result = (DataObject[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        DataObject[col] = DataObject[col].astype(np.uint8)
                    elif mx < 65535:
                        DataObject[col] = DataObject[col].astype(np.uint16)
                    elif mx < 4294967295:
                        DataObject[col] = DataObject[col].astype(np.uint32)
                    else:
                        DataObject[col] = DataObject[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        DataObject[col] = DataObject[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        DataObject[col] = DataObject[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        DataObject[col] = DataObject[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        DataObject[col] = DataObject[col].astype(np.int64)    
            else:
                DataObject[col] = DataObject[col].astype(np.float32)
            
            #print("Changed DataType : ",DataObject[col].dtype)
            Inference['ModifiedDataType'].append(DataObject[col].dtype)
        else:
            Inference['ModifiedDataType'].append(DataObject[col].dtype)                
    #print("******************************")
    
    mem_usg = DataObject.memory_usage().sum() / 1024**2 
    print("Changed Usage : ",mem_usg," MB")
    print("Reduction : ",100*mem_usg/start_mem_usg,"%")
    print("*******************************************")
    #print("Columns With Missing Values [ Formula Applied - 'DataObject['column_name'].min() -1' ] : ") 
    #print(NullDataColumns)
    #print("******************************")
    Inference = pd.DataFrame(Inference, columns = ['ColumnName', 'OriginalDataType', 'ModifiedDataType', 'HasNullData']).sort_values('HasNullData',ascending=False)
    return DataObject, Inference

class FeatureSelection:
    
   def BackwardElimination(self, DataObject, Results, SignificanceLevel=None, FigureSize=None):
    if SignificanceLevel is None:
        SignificanceLevel = 5/100
    NoOfColumns = DataObject.shape[1]
    import statsmodels.formula.api as sm
    import pandas as pd
    import numpy as np    
    DataObject = np.append(arr = np.ones((DataObject.shape[0],1)).astype(int), values = DataObject, axis = 1)
    DataObjectColumns = list(range(0, DataObject.shape[1]))
    DataObjectColumnsDecision = list(range(0, DataObject.shape[1]))
    Inference = {'SlNo': [],'ColumnNo': [],'PValue': [],'Skew': [],'Kurtosis': [],'R-Squared': [],'Adj-R-Squared': [],'F-Statistic': []}
    LastARSValue = 0.000
    for Column in range(0,NoOfColumns):
        #print(DataObjectColumnsDecision)
        #print('|||||||||||||')
        DataObjectTemp = DataObject[:, DataObjectColumns]
        OLS = sm.OLS(Results,DataObjectTemp).fit()
        OutputI = (OLS.summary2().tables[0])
        OutputII = (OLS.summary2().tables[1])    
        #print(OLS.summary())
        OutputIII = (OLS.summary2().tables[2]) 
        OutputII['Column'] = DataObjectColumns 
        HigherPValues = OutputII.loc[(OutputII['P>|t|'] > SignificanceLevel) & (OutputII['Column'] > 0)].sort_values('P>|t|',ascending=False)
        if(HigherPValues.shape[0] > 0):
            Inference['SlNo'].append(Column+1)
            Inference['ColumnNo'].append(HigherPValues.iloc[0, 6]-1)
            Inference['PValue'].append(HigherPValues.iloc[0, 3])
            Inference['Skew'].append(OutputIII.iloc[2, 1])
            Inference['Kurtosis'].append(OutputIII.iloc[3, 1])  
            Inference['R-Squared'].append(OutputI.iloc[6, 1])
            Inference['Adj-R-Squared'].append(OutputI.iloc[0, 3])
            Inference['F-Statistic'].append(OutputI.iloc[4, 3]) 
            if(Column > 0):
                if("{0:.3f}".format(float(OutputI.iloc[0, 3])) > LastARSValue):
                    DataObjectColumnsDecision.remove(HigherPValues.iloc[0, 6])
                    #print(HigherPValues.iloc[0, 6])
                    #print('----------')  
            else:
                DataObjectColumnsDecision.remove(HigherPValues.iloc[0, 6])
                #print(HigherPValues.iloc[0, 6])
                #print('^^^^^^^^^^^^^')                   
            LastARSValue = "{0:.3f}".format(float(OutputI.iloc[0, 3])) 
            #print(LastARSValue)
            #print('////////////')   
            #print('')
            #print('')
            DataObjectColumns.remove(HigherPValues.iloc[0, 6])
        else:
            HigherPValues = OutputII.loc[(OutputII['Column'] > 0)].sort_values('P>|t|',ascending=False)
            if(HigherPValues.shape[0] > 0):
                Inference['SlNo'].append(Column+1)
                Inference['ColumnNo'].append(HigherPValues.iloc[0, 6]-1)
                Inference['PValue'].append(HigherPValues.iloc[0, 3])
                Inference['Skew'].append(OutputIII.iloc[2, 1])
                Inference['Kurtosis'].append(OutputIII.iloc[3, 1])  
                Inference['R-Squared'].append(OutputI.iloc[6, 1])
                Inference['Adj-R-Squared'].append(OutputI.iloc[0, 3])
                Inference['F-Statistic'].append(OutputI.iloc[4, 3])      
            break
    Inference = pd.DataFrame(Inference, columns = ['SlNo', 'ColumnNo', 'PValue', 'Skew', 'Kurtosis', 'R-Squared', 'Adj-R-Squared', 'F-Statistic'])
    DataObjectColumns.remove(0)
    DataObjectColumns = [C-1 for C in DataObjectColumns] 
    DataObjectColumnsDecision.remove(0)
    DataObjectColumnsDecision = [C-1 for C in DataObjectColumnsDecision]
    Inference = Inference.astype(float)
    import matplotlib.pyplot as plt
    if FigureSize is not None:
        plt.figure(figsize=FigureSize)
    plt.plot( 'SlNo', 'R-Squared', data=Inference, marker='o', markerfacecolor='blue', color='orange', linewidth=2, label="R-Squared")
    plt.plot( 'SlNo', 'Adj-R-Squared', data=Inference, marker='o', markerfacecolor='blue', color='green', linewidth=2, linestyle='dashed', label="Adj-R-Squared")
    plt.legend()
    plt.title('Backward Elimination ( Dimension / PValue )')
    plt.xlabel('Iteration')
    plt.ylabel('COD')
    plt.grid()
    plt.xticks(np.arange(min(Inference['SlNo']), max(Inference['SlNo'])+1, 1.0))
    for Row in range(0,Inference.shape[0]):
        plt.text(Inference.iloc[Row, 0], Inference.iloc[Row, 5],
                 ('  ('+str(int(Inference.iloc[Row, 1]))+' / '+"{0:.3f}".format(Inference.iloc[Row, 2])+')'),verticalalignment='bottom')
    plt.show()        
    return DataObjectColumnsDecision,Inference
    
#class Models:
    
#   def Regression(self, DataObject, Results, Ratio=None, Scale=None, KFold=None, FigureSize=None):

#class Visualization:
