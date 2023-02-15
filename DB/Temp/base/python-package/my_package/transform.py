# Databricks notebook source
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Pipeline


# Inherit Transformer
class WordsRemover(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):

    # Set up Parameter names and defaults
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, words=None):
        super(Transformer, self).__init__()
        self.words = Param(self, "words", "Words to remove")
        self._setDefault(words=[])
        self._setDefault(inputCol="features")
        self._setDefault(outputCol="cleansed")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    # This is a standard function for setting more parameters at once
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, words=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # setting the words
    def setWords(self, value):
        self._paramMap[self.words] = value
        return self

    # setting the words
    def getWords(self):
        return self.getOrDefault(self.words)

    # Business Logic comes here
    def _transform(self, dataframe):
        words = self.getWords()

        def f(s):
            ret = s
            for w in words:
                ret = ret.replace(w, "")
            return ret

        out_col = self.getOutputCol()
        in_col = dataframe[self.getInputCol()]
        return dataframe.withColumn(out_col, udf(f, StringType())(in_col))
