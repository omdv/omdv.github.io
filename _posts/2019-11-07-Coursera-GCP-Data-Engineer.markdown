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

2002 - Google File System
2004 - MapReduce, which led to Hadoop
2006 - YouTube and high data volume led to BigTable, which inspired other no-sql databases
2008 - Dremel as a substitution of MapReduce
2013 - PubSub and DataFlow
2015 - Tensorflow
2017 - TPUs
2018 - AutoML


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

### Course 1.3 - Predict visitor purchases with BigQuery ML
