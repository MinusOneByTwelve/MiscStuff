# Databricks notebook source
remote_files = ["/COVID/", "/COVID/coronavirusdataset/", "/COVID/coronavirusdataset/Time.csv", "/COVID/coronavirusdataset/coronavirusdataset-readme.md", "/airbnb/", "/airbnb/README.md", "/airbnb/sf-listings/", "/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", "/airbnb/sf-listings/models/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/data/_committed_6035389993324123394", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/data/_started_6035389993324123394", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/data/part-00000-tid-6035389993324123394-30d71b7d-d7a6-46fe-bc1c-3130273ee4dd-17276-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/00_strIdx_1513d2eb26de/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/data/_committed_2079527968196944861", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/data/_started_2079527968196944861", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/data/part-00000-tid-2079527968196944861-4198dee7-06d3-463f-ad00-d431404b15a3-17279-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/01_oneHotEncoder_4953d1936386/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/data/_committed_8087729930989222520", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/data/_started_8087729930989222520", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/data/part-00000-tid-8087729930989222520-7ef6ed8a-2223-4493-9f1d-15d029d8647f-17282-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/02_strIdx_a4c353763d72/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/data/_committed_1737519085480631885", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/data/_started_1737519085480631885", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/data/part-00000-tid-1737519085480631885-1c59b10c-e8c0-4bde-96d1-e7214efe026a-17285-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/03_oneHotEncoder_33250a9607f8/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/data/_committed_4602327655585755686", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/data/_started_4602327655585755686", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/data/part-00000-tid-4602327655585755686-3253c2d0-a4d1-4093-9b3f-e940c39c6777-17288-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/04_strIdx_7920f98db082/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/data/_committed_2553817499036123685", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/data/_started_2553817499036123685", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/data/part-00000-tid-2553817499036123685-7abaafc7-1d70-4df2-8d61-e8a59a7f8b76-17291-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/05_oneHotEncoder_6a45e1370f07/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/data/_committed_2665209346957440496", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/data/_started_2665209346957440496", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/data/part-00000-tid-2665209346957440496-35dc3bce-0061-41bf-b291-95aa9c72b52c-17294-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/06_strIdx_bc9668ab9a4f/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/data/_committed_1091098354965480121", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/data/_started_1091098354965480121", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/data/part-00000-tid-1091098354965480121-1033bcf7-e105-49dd-926d-cb5c6f96cc53-17297-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/07_oneHotEncoder_e7c6f7ce28e0/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/data/_committed_1129441435070285754", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/data/_started_1129441435070285754", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/data/part-00000-tid-1129441435070285754-bb3b5487-9cdb-412a-a5f0-43ce8b86469a-17300-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/08_strIdx_e4725aeddbfc/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/data/_committed_714447836921176592", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/data/_started_714447836921176592", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/data/part-00000-tid-714447836921176592-c7e892cb-771c-466c-b49d-73ba990d0ccc-17303-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/09_oneHotEncoder_099bc6a266be/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/data/_committed_448960854689203664", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/data/_started_448960854689203664", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/data/part-00000-tid-448960854689203664-90ae6d0d-5755-4256-96fe-cbb1d09796ce-17306-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/10_strIdx_c83401f53fb2/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/data/_committed_4177592625104903462", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/data/_started_4177592625104903462", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/data/part-00000-tid-4177592625104903462-db1a6343-8c02-4d30-a71f-266b090ae407-17309-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/11_oneHotEncoder_5aebd3179a5a/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/data/_committed_7341023940658473476", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/data/_started_7341023940658473476", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/data/part-00000-tid-7341023940658473476-277f87e5-afbe-4082-a33d-a8d980efb00e-17312-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/12_strIdx_9000b1835d75/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/data/_committed_1527229736105433191", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/data/_started_1527229736105433191", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/data/part-00000-tid-1527229736105433191-d2dc6956-b66f-4e8e-935c-c40d67629767-17315-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/13_oneHotEncoder_d2d3e572e18f/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/data/_committed_3075108839144722058", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/data/_started_3075108839144722058", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/data/part-00000-tid-3075108839144722058-2b8d473a-c0af-4ba4-adc4-d9d2b3fd4fc5-17318-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/14_strIdx_de78b4034b53/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/data/_committed_1027951254404413455", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/data/_started_1027951254404413455", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/data/part-00000-tid-1027951254404413455-4c4a3504-c154-4bb4-a8c9-e3de7cc8b251-17321-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/15_oneHotEncoder_56899e2fb6e9/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/16_vecAssembler_85d2cff6b603/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/16_vecAssembler_85d2cff6b603/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/16_vecAssembler_85d2cff6b603/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/16_vecAssembler_85d2cff6b603/metadata/part-00000", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/data/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/data/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/data/_committed_841328777751729849", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/data/_started_841328777751729849", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/data/part-00000-tid-841328777751729849-29fd10b4-5b50-43c1-8bcb-ae02c5457276-17325-1-c000.snappy.parquet", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/metadata/", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/metadata/_SUCCESS", "/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model/stages/17_linReg_22bc9e0ff0b1/metadata/part-00000", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/_SUCCESS", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/_committed_773940399323573814", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/_started_773940399323573814", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00000-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16947-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00001-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16948-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00002-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16949-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00003-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16950-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00004-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16951-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00005-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16952-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00006-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16953-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00007-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16954-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00008-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16955-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00009-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16956-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00010-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16957-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00011-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16958-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00012-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16959-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00013-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16960-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00014-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16961-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00015-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16962-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00016-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16963-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00017-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16964-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00018-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16965-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00019-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16966-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00020-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16967-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00021-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16968-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00022-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16969-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00023-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16970-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00024-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16971-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00025-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16972-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00026-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16973-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00027-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16974-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00028-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16975-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00029-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16976-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00030-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16977-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00031-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16978-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00032-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16979-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00033-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16980-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00034-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16981-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00035-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16982-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00036-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16983-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00037-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16984-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00038-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16985-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00039-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16986-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00040-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16987-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00041-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16988-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00042-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16989-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00043-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16990-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00044-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16991-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00045-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16992-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00046-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16993-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00047-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16994-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00048-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16995-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00049-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16996-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00050-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16997-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00051-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16998-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00052-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-16999-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00053-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17000-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00054-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17001-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00055-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17002-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00056-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17003-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00057-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17004-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00058-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17005-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00059-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17006-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00060-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17007-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00061-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17008-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00062-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17009-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00063-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17010-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00064-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17011-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00065-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17012-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00066-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17013-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00067-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17014-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00068-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17015-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00069-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17016-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00070-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17017-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00071-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17018-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00072-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17019-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00073-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17020-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00074-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17021-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00075-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17022-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00076-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17023-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00077-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17024-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00078-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17025-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00079-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17026-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00080-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17027-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00081-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17028-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00082-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17029-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00083-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17030-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00084-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17031-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00085-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17032-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00086-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17033-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00087-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17034-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00088-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17035-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00089-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17036-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00090-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17037-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00091-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17038-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00092-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17039-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00093-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17040-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00094-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17041-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00095-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17042-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00096-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17043-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00097-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17044-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00098-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17045-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/part-00099-tid-773940399323573814-405a9fc3-d671-450e-b78b-78ebc9e2fada-17046-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/_delta_log/", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/_delta_log/00000000000000000000.crc", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/_delta_log/00000000000000000000.json", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/part-00000-3787af23-a2e0-4036-973a-1190fba41c74-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/part-00001-2858da66-0385-4f5e-9afe-6d91e4c2bb35-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/part-00002-86b7ae15-b8d4-44d2-9107-24cad47c3a42-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/part-00003-96bfa877-4f48-4ace-8c5a-a886ff62dec2-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/_SUCCESS", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/_committed_4320459746949313749", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/_started_4320459746949313749", "/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/part-00000-tid-4320459746949313749-5c3d407c-c844-4016-97ad-2edec446aa62-6688-1-c000.snappy.parquet", "/airbnb/sf-listings/sf-listings-2019-03-06.csv", "/dataframes/", "/dataframes/README.md", "/dataframes/people-with-dups.txt"]

