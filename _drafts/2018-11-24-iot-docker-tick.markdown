## Creating Grafana dashboard

We will use Grafana templating and variables functionality. Since we want to see the sensor time series for every available host in InfluxDB we need to create dashboard variables first. Make sure to enable "Refresh".

![Grafana variables]({{ site.url }}/assets/2018-11-24-grafana-variables.png)

Next step is to create a chart and create a query using our defined variable. The query needs to return multiple series, so we will use `GROUP BY $host`.

![Grafana variables]({{ site.url }}/assets/2018-11-24-grafana-plot-setup.png)

This is the plot for 5 edge devices running a sine model with 10s update rate.

![Grafana plot 10s]({{ site.url }}/assets/2018-11-24-grafana-plot-10s.png)

## Enabling Kapacitor

Kapacitor is configured to run as a part of docker stack. Chronograf is best to use as a configuration tool for capacitor to avoid messing with multiple commands inside a running containers shell.

```
stream
    // Select the measurement
    |from()
        .measurement('edge')
	// Create new variable
    |eval(lambda: "sensor" > 5)
    	.as('gt5')
	// Useful part to debug TICK scripts
    |log()
    // Write the data to InfluxDB
    |influxDBOut()
        .database('iotdb')
        .retentionPolicy('autogen')
        .measurement('alarms')
        .tag('kapacitor', 'true')
```

## Kapacitor and UDF

Unfortunately python3 support for kapacitor is currently [lacking](https://github.com/influxdata/kapacitor/pull/1355). I tried to apply the mentioned patches however failed to get a working solution.

## Managing execution times
Choosing timeout in go client