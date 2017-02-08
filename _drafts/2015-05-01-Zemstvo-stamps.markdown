---
layout: post
title:  "Zemstvo stamps map"
categories: [data mining, visualization, d3js]
---
There was a question posted on [stackoverflow][stack] on classification issues in WEKA. The datasets are available via [web-archive][datasets]. Since *.csv files were not available I downloaded *.ARFF and converted them to *.csv manually, as the format appeared to be quite simple. This is the list of features available to be placed in first row to construct the dataframe with column names.

{% highlight python %}
duration,protocol_type,service ,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,class
{% endhighlight %}

I read both training and test datasets into pandas DataFrame. Merge them into one to do the feature processing and later split using the original ratio. Three features which had strings as values were factorized, as well as the outcome variable "class".

The code is below. Libraries used are numpy, pandas, scikit-learn and matplotlib. One can use Anaconda python distribution which 

[stack]: http://stackoverflow.com/questions/35882933/classfication-accuracy-on-weka/35883064#35883064
[datasets]: https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/