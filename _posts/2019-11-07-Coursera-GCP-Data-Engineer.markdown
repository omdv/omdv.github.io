---
title: Coursera - Data Engineering with GCP
layout: post
tags: [coursera, machine learning, data, google, notes]
description: Notes for the Coursera Data Engineer Specialization.
---

## Course 1 - Google Cloud Platform Big Data and Machine Learning Fundamentals

According to McKinsey - by 2020 there will be 5bln connected devices, with amount of data doubling every two years. Also it is estimated that only 1% of this data is currently analyzed.

Big Data Challenges are:
- Migrating existing data workloads (ex. Hadoop, Spark)
- Analyzing large datasets at scale (note "large" means Tbs or Pbs)
- Building scalable streaming data pipelines
- Applying machine learning models to your data


### Course 1.1 - Introduction to GCP

Introduction to GCP which is behind 7 cloud products with 1bn monthly users. And GCP itself is responsible for hosting another ~1bn of end-users with GCP customer products.

Three key layers:
- base layer is Security
- 2nd layer is Compute, Storage and Networking
- 3rd layer is Big Data and ML products to abstract away the BD and ML infrastructure

#### Compute

From 2018 estimates, roughly 1.2 billion photos and videos are uploaded to the Google Photos service everyday. That is 13 plus petabytes of photo data in total. For Youtube - 60Pb of data every hour.

Moore's law is not valid since ~2003-2005, now the compute power doubles only every ~20yrs or so. One solution is [ASIC](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit), and Google's Tensor Processing Unit is one of the examples of ASIC. TPUs brought 10x increase in ML training/iterations of models.

GCP demo covering:
- creating and preparing compute VM
- creating bucket
- copying processed data to the bucket and stopping VM
- changing permissions of files in the bucket to public
- viewing static site on storage.googleapis


#### Storage

Four classes of storage:
- Multi-regional
- Regional
- Nearline
- Coldline

Resource hierarchy:
1. Resources
2. Projects, provide logic grouping of resources
3. Folders, can combine projects
4. Organization, group node for the enterprise-level of policies

#### Networking

Google private network connecting GCP data centers carries ~40% of entire internet traffic. Fast network allows separating computations and data across multiple data centers w/o compromising communications between resources.

#### Security

The last piece of basic insfrastructure. In the on-premise deployments you are responsible for security across all layers, including networking, hardware, storage, deployment, etc. As you move towards managed services you can offload these concerns to GCP, so you are left with security on Content, Access Policies and Usage.

In-transit encryption, encryption at rest and distributed for access/storage. BigQuery security controls as an example.

#### Evolution of GCP

- 2002 - Google File System
- 2004 - MapReduce, which led to Hadoop
- 2006 - YouTube and high data volume led to BigTable, which inspired other no-sql databases
- 2008 - Dremel as a substitution of MapReduce
- 2013 - PubSub and DataFlow
- 2015 - Tensorflow
- 2017 - TPUs
- 2018 - AutoML


Choosing the right approach/product. Starting from IaaS in terms of Compute, GKE, App Engine as PaaS and Cloud Functions and FaaS.

Key roles in Big Data companies:
- Data Engineers to build pipelines
- Decision Makers for investment
- Data Analysts to explore data for insights
- Statisticians
- Applied ML Engineers
- Data Scientists
- Analytics Managers

#### Links
1. [GCP Blog](https://cloud.google.com/blog/products)
2. [GCP Big Data products](https://cloud.google.com/products/big-data/)
3. [BigQuery SQL](https://cloud.google.com/bigquery/docs/reference/standard-sql/enabling-standard-sql)
4. [QwikLabs Data Analyst SQL Quest](https://www.qwiklabs.com/quests/55)

### Course 1.2 - Recommending System using Spark and Cloud SQL

Product Recommendations is one of the most common business tasks. Example of application - housing recommendation system on premise. Data is stored in Hadoop. There is an ML model running as SparkML job on data in Hadoop _daily_ with results of the recommendation stored in MySQL. This is all running on premise. Now we are interested in moving to GCP. We'll move the SparkML job as DataProc with results stored in Cloud SQL.

Typical patterns for choosing the right database solution:
- "Cloud Storage" for blob file storage
- "Cloud SQL" for gigabytes-level SQL storage on the cloud. If data > Gbs and > 1 table -> "Cloud Spanner"
- "Cloud Datastore" for structured data from App Engine, transactional no-SQL.
- "BigTable" for non-transactional, high throughput flat data (sensors)
- "BigQuery" for data warehousing

Cloud Dataproc provides a managed service for creating Hadoop and Spark clusters, and managing data processing workloads.

Open-source ecosystem running on Hadoop:
1. Hadoop is canonical MapReduce framework
2. Pig is a scripting language to create MapReduce jobs
3. Hive is data warehousing system
4. Spark is a fast interactive general-purpose framework to run analytics, leveraging in-memory processing

Demo starting a Spark job on Hadoop cluster using DataProc in less than 10min.

Lab covering the following:
- Creating Cloud SQL cluster and database with three tables
- Importing data to Cloud SQL from buckets and running queries
- Creating DataProc cluster
- Submitting a PySpark job running on Cloud SQL cluster

#### Links
1. [Migrating Hadoop to GCP](https://cloud.google.com/solutions/migration/hadoop/hadoop-gcp-migration-overview)
2. [Dataproc documentation](https://cloud.google.com/dataproc/)

### Course 1.3 - Predict visitor purchases with BigQuery ML

BigQuery as a petabyte-scale analytics warehouse. BQ is actuall two services: a fast SQL Query Engine and fully managed data storage.

BQ features:
1. Serverless
2. Flexible pricing
3. Data encryption and security
4. Geospatial data types
5. Foundation for BI and AI

Typical BQ solution architecture: batch (cloud storage) and streaming (pub/sub) pipelines feeding the BQ via  Dataflow. BQ can then expose data through DataStudio, Tableau, Qlik, Looker and even Google spreadsheets.

Demo of querying github 2M repos dataset to determine tabs vs spaces ratio by programming language. One SQL script, executed over 155Gb of data in 13sec with ~50min of actual nodes work time.

BQ storage hierarchy is Datasets consisting zero or more tables, each consisting columns. Columns are compressed and stored in Colossus filesystem.

Demo of exploring public bike share dataset in SFO area with just SQL. `CREATE or REPLACE` to store results in a table or a view, which provides logic only.

Misc quick notes:
- Cloud Dataprep (partnership with Trifecta) to provide visualizations to evaluate the quality of data in BQ.
- Scheduling BQ SQL jobs with @run_time
- Use BQ queries on federated data sources (GCS, Google drive, CSV)
- Streaming data into BQ through API or DataFlow
- BQ natively supports structs and arrays for the case of data normalization
- BQ natively supports GIS, BQ Geo Viz also gives a free GIS visualization tool

#### BigQuery ML

Simple decision tree for choosing the right ML models:
- Linear regression for time-series forecast
- Logistic regression (single or multi-class) for classification
- Matrix factorization for recommendation
- Clustering for unsupervised

Built-in functions for training and prediction inside SQL. BQ ML supports basic features, such as auto-tuning learning rate and auto-splitting into training and test datasets, but also supports more advanced like regularization, different training/test splits, custom learning rates, etc. Supports linear regression and logistic regressions. Supports different metrics, inspecting weights, etc.

E2E BQML process:
1. ETL data into BQ
2. Preprocess features
3. Create model
4. Evaluate model
5. Use model for predictions

Alias a label column as `label` or specify it in options.

#### Lab on BQ ML

Example of model creation based on Google Analytics dataset with two features and one label:

```
CREATE OR REPLACE MODEL `ecommerce.classification_model`
OPTIONS
(
model_type='logistic_reg',
labels = ['will_buy_on_return_visit']
)
AS

#standardSQL
SELECT
  * EXCEPT(fullVisitorId)
FROM

  # features
  (SELECT
    fullVisitorId,
    IFNULL(totals.bounces, 0) AS bounces,
    IFNULL(totals.timeOnSite, 0) AS time_on_site
  FROM
    `data-to-insights.ecommerce.web_analytics`
  WHERE
    totals.newVisits = 1
    AND date BETWEEN '20160801' AND '20170430') # train on first 9 months
  JOIN
  (SELECT
    fullvisitorid,
    IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
  FROM
      `data-to-insights.ecommerce.web_analytics`
  GROUP BY fullvisitorid)
  USING (fullVisitorId)
;
```

### Course 1.4 - Create Streaming Data Pipelines with Cloud Pub/sub and Cloud Dataflow

First challenge - handling large volumes of streaming data, not coming from the single source. Solution is - Cloud Pub/Sub.

Cloud Pub/Sub:
- at-least-once delivery
- exactly-once processing
- no provisioning
- open APIs
- global by default
- e2e encryption

Common architecture is: Ingest -> Cloud Pub/Sub for storing -> Dataflow for processing -> BQ or SQL for Storing and Analyzing -> Explore and Visualize

During streaming pipelines design the Data Engineer needs to address two questions:
- Pipeline design (Apache Beam)
- Pipeline implementation (Dataflow)

Apache Beam is Unified, Portable, Extensible. Has a syntax very similar to scikit-learn pipelines. You can write a code in Java, Go, python and deploy in Runner (Spark, Dataflow). Dataflow is scalable, serverless and has Stackdriver integrated in it. After deploying a pipeline in Dataflow it will automatically optimize it, scale up workers, heal workers, connect to many different sinks.

DataStudio can work with multiple datasources. Some ramifications - anyone who can see the dashboard can potentially see the whole datasources. Anyone who can edit the dashboard can see all fields.

Some dashboard tips:
- start with KPIs at the top
- below have sections with details, starting with the question "What are the sales channels?" to grab user attention
- you can fix time filters to some specific periods

##### Lab
Lab on setting up a Dataflow pipeline connecting to the public pub/sub topic on NY taxi rides, outputting the results to BQ. Creating a SQL query on BQ to output the summary of rides with passengers, rides and total revenue. Connecting Datastudio for creation of the streaming dashboard.

#### Links
1. [Cloud Pub/Sub documentation](https://cloud.google.com/pubsub/docs/)
2. [Cloud Dataflow documentation](https://cloud.google.com/dataflow/docs/)
3. [Data Studio documentation](https://developers.google.com/datastudio/)

### Course 1.5 - Classify Images with Pre-Built Models using Vision API and Cloud AutoML

Examples of business applications:
- pricing cars based on images
- NPL to route emails to correct support
- image recognition to mark inappropriate content in images
- Dialogflor for new shopping experience
- predicting empty or full cargo ships based on the satellite image

Approaches to ML according to GCP:
1. Pre-Built AI products
2. Custom models on top of AutoML
3. New models entirely with training

Vision API: cloud.google.com/vision

By 2021 enterpises will use 50% more chatbots - GCP Dialogflow.

**Text classification problem done in three ways using GCP**

Predicting which publication the article came from, based on just 5 first letters in the title.
1. BQ ML: sql processing to create training dataset. Use BQ ML to train, evaluate and predict
2. AutoML Natural Language: takes few hours to train
3. Custom model using ML Engine Jupyter notebooks. Jupyter magic to use BQ inside notebooks.

#### Links
1. [GCP Data Analyst Training](https://github.com/GoogleCloudPlatform/training-data-analyst)

## Course 2 - Leveraging Unstructured Data

Unstructured data is seen as "digital exhaust". Big, expensive to process.

### Course 2.1 - Introduction to Dataproc

Data you collect, but don't analyze, because it is too hard, usually. This is the data, which can be described by 3Vs: veracity, velocity, and volume.

Structured data has schema. Unstructured data is a relative term. It may have schema, but not suitable for analysis. About 90% of enterprise data is unstructured.

Dataproc is GCP implementation of Hadoop.

Horizontal scaling (cloud) vs vertical scaling (mainframe).

In early 2000 Google realized the need to index all web. In 2004 Google proposed MapReduce. Map splits BigData and maps it to some structure and Reduce reduces to some meaningful results. Later the Google File System was created to manage distributed storage. Apache developed an OSS alternative, called Hadoop, with Hadoop File System. Pig, Hive and Spark are parts of the ecosystem built on top of Hadoop.

Apache Spark is an important innovation, which allows managing multiple various workflows (say sensor processing vs image processing) and map it to available Hadoop resources. Spark follows the Declarative paradigm, i.e. it decides how to do things, based on What you want.

... marketing speech on why GCP is better than on-premise cluster ...

Options for creating Hadoop cluster:
- on-premise
- vendor-managed
- bdutil (OSS utility to create clusters using VMs) - DEPRECATED in favor of Dataproc
- Dataproc

Dataproc on average takes 90sec to provision from start to beginning of processing.

Dataproc customization:
- single node cluster (good for experimentation)
- standard (one master node)
- high-availability (three master nodes)

Don't use HDFS for cloud storage. Dataproc cluster should be stateless, so you can turn it on/off as needed.

#### Lab - Creating Hadoop cluster and exploring it

Create a VM to control the cluster. SSH into a vm. Setup environment variables (not sure why, as all provisioning was done in WebUI). Setup and start the 3-worker node cluster. Create firewall rule using a network tag. Make sure that cluster master has a right network tag. Connect to its public IP to see Hadoop cluster status.

#### Choosing the right size of your VMs

Pick the number of vCPUs and RAM to match the job requirements. GCP WebUI allows to copy/paste the custom code for the customized VM.

Preemptible VMs are up to 80% cheaper than standard. Use PVMs only for processing. Use it for non-critical processing and in large clusters. Rule of thumb - anything over 50/50 ratio of PVMs vs standard may have diminishing returns. Dataproc manages adding and removing PVMs automatically.


### Course 2.2 - Running Dataproc jobs

Dataproc includes some of the most common Hadoop frameworks: Hive, Pig and Spark.
Hive is declarative, Pig is imperative.

#### Lab - Hive and Pig jobs

Example of Pig job:
```
rmf /GroupedByType
x1 = load '/pet-details' using PigStorage(',') as (Type:chararray,Name:chararray,Breed:chararray,Color:chararray,Weight:int);
x2 = filter x1 by Type != 'Type';
x3 = foreach x2 generate Type, Name, Breed, Color, Weight, Weight / 2.24 as Kilos:float;
x4 = filter x3 by LOWER(Color) == 'black' or LOWER(Color) == 'white';
x5 = group x4 by Type;
store x5 into '/GroupedByType';
```

Pig provides SQL primitives similar to Hive, but in a more flexible scripting language format. Pig can also deal with semi-structured data, such as data having partial schemas, or for which the schema is not yet known. For this reason it is sometimes used for Extract Transform Load (ETL). It generates Java MapReduce jobs. Pig is not designed to deal with unstructured data.

#### Separation of storage and compute resources