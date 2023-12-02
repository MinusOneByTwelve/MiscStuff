# Databricks notebook source
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, when

class MissingValueIndicator(Transformer, HasInputCols, HasOutputCols, 
                            DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, modeDict=None):
        super(MissingValueIndicator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)  

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, modeDict=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)           

    def _transform(self, df):
        """
        Returns a DataFrame containing inputCols and their binary indicator columns

        Parameters
        ----------
        dataset: DataFrame
            The original dataframe with categorical column as input column

        Returns
        -------
        Dataframe with imputed mode: DataFrame    
        """      
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()

        for input_col, output_col in zip(input_cols, output_cols):
              df = df.withColumn(
                  output_col, 
                  when(col(input_col).isNull(), 1.0).otherwise(0.0)
              )
        return df

