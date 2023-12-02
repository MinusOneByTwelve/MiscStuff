import pytest
from MultiTaskSqlJob import ComplexBusinessLogic

@pytest.mark.usefixtures("spark_session")
def Test_ComplexBusinessLogic(spark_session):
    test_df = spark_session.createDataFrame(
        [
            ('Expenses', 'ADMIN', 5),
            ('Expenses', 'IT', 50),
            ('Expenses', 'IT', 20),
            ('Income', 'SALES', 1000)
        ],
        ['Category', 'Department', 'Percentage']
    )
    new_df = ComplexBusinessLogic(test_df)
    assert new_df.count() == 1
    assert new_df.toPandas().to_dict('list')['FinalPerc'][0] == 70
