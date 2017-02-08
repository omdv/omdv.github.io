---
layout: post
title:  "Painters Timeline - Part 1: Data Mining"
date:   2016-02-24 23:52:26 -0700
categories: [art history, data mining, data science]
---
I started reading ["The Story of Art"][story-art] recently - a great classic piece if you have any interest in the topic. As I am reading it I naturally started wondering about possible connections between artists and painters of different generations and artistic movements, trying to visualize the evolution. Somewhat surprisingly I didn't find a good resource trying to analyze this using the latest data science techniques, so I decided to give it a try. I have somewhat vague idea of the final result - perhaps some kind of a visualization where you can see painters timeline, associations with artistic movements, maybe connections between each other and so on. So this is going to be an exploratory study, consisting of few parts. I'll start with data mining and some exploratory analysis and hopefully will reach a point when I can apply some Machine Learning techniques. So, enough for motivation, let's start.

# Setup

I will use [Anaconda Python][anaconda] distribution. It comes with 400 libraries, including those major for data science. The included Spyder is a good IDE, which in combination with IPython support provides a Matlab-like environment supporting a quick prototyping.


# Getting Data

Wikipedia is a natural starting point for a list of painters. There is a manually maintained [alphabetic list][wiki-painters-list], consisting of more than 3,000 painters. It appears to be a good starting point. I will use `requests` library to download, `lxlm` to parse it and `ascii_uppercase` list to iterate over all urls.

{% highlight python %}
import requests
from lxml import html
from string import ascii_uppercase

#download the list of painters from wiki artist list
#save to a given filename
#output the number of parse painters
def download_artists_list(filename):
  painters = []

  for char in ascii_uppercase:
      req = 'http://en.wikipedia.org/wiki/'
      req += 'List_of_painters_by_name_beginning_with_%22'+char+'%22'
      page = requests.get(req)
      tree = html.fromstring(page.content)
  
      #This will create a list of painters
      parsed = tree.xpath('//div[@id="mw-content-text"'+ 
          'or contains(@class,"div-col")]/ul/li/a[1]')
      if not parsed:
          parsed = tree.xpath('//div[@id="mw-content-text"'+\
              'or contains(@class,"div-col")]/div/ul/li/a[1]')
  
      for parse in parsed:
          painter = {}
          
          href=parse.attrib['href']
          #check if url is on wiki, otherwise - ignore
          if re.match("/wiki",href):
              painter['href']=href
              painter['name']=href.split('/')[2].replace("_"," ")
              painters.append(painter)

  #pickle painters
  with open(filename, 'w') as fp:
      json.dump(painters,fp)
  return len(painters)
{% endhighlight %}

The function above should be pretty trivial. I iterate over upper case letters, create a request and parse the response using `lxml.html` parser. 

Coming up with a proper `xpath` to get the information you want is the most important piece of the puzzle. I highly recommend to use Chrome or Firefox Developer tools to do this. Unfortunately in many cases you are at mercy of the creator of the page and may need to use the combination of different paths. In this particular case I am getting all `<a>` tags in the corresponding lists under `<div id="mw-content-text">`. Two lists starting with "S" and "T" use two columns and as a result I have to look for another `<div class="div-col">` as well. Finally in several cases these lists are put under extra div tag, so there is a condition to use this xpath if the other one fails to find anything.

Once I have the list of all `<a>` tags on the given page I extract it's `href` attribute and check whether it's linked to a page on wikipedia, as I don't want to worry about different page formats to parse later. Out of more than 3,000 records only 3-4 were external. Finally I split the link to save the name and save the dataset as json file. 

The function above generated 3148 records of artists, which so far consists only of corresponding wikipedia links. The next step is to download these wiki pages.

I am going to use pandas DataFrame to manipulate all datasets with the help [pandas.Dataframe.apply()][pd-apply]. I'll cover it later, but the main thing to note is that in this case we need a function to handle only one row of DataFrame, as the iteration will be handled effectively by `apply()`.

Let's create the function to parse individual wiki pages.

{% highlight python %}
#get the content from wiki
#output - saved files and content
def get_content_from_page(href):
  print href
  result = {}
  
  #request and response
  page = requests.get('http://en.wikipedia.org'+href)
  tree = html.fromstring(page.content)
  
  #Getting abstract
  paragraphs = tree.xpath('//*[@id="mw-content-text"]'+
      '/p[following::*[@id="toc"]]')
  #Get text content
  content = []
  for para in paragraphs:
      text = unicode(para.text_content()).encode(sys.stdout.encoding,'replace')
      content.append(text)
  result['abstract'] = ' '.join(content)
  
  #Getting rest of content or full content if abstract is empty and no TOC
  if result['abstract'] == '':
      paragraphs = tree.xpath('//*[@id="mw-content-text"]/p')
  else:
      paragraphs = tree.xpath('//*[@id="mw-content-text"]'+
          '/p[preceding::*[@id="toc"]]')
  #Get text content    
  content = []
  for para in paragraphs:
      text = unicode(para.text_content()).encode(sys.stdout.encoding,'replace')
      content.append(text)
  result['content'] = ' '.join(content)
  
  #Check if there is infobox and parse
  table = tree.xpath('//table[contains(@class,"infobox")'
      'or contains(@class,"biography")]')
  bio = {}    
  if table:
      rows = table[0].findall('tr')
      for row in rows:
          th = row.findall('th')
          td = row.findall('td')
          if (th and td):
              bio[th[0].text_content()] = td[0].text_content().replace('\n',' ')
      result['biography']=bio
  else:
      result['biography']=None
  
  #dump result to a separate file
  filename = unicode(href.split('/')[2]).encode(sys.stdout.encoding, 'replace')
  filename = re.sub("\.","",filename)
  with open('painters_new/'+filename, 'w') as fp:
      json.dump(result,fp)
  
  return 0
{% endhighlight %}

The function above has few blocks. The first one is just getting a response and applying an html parser. Then I handle two possible structures of a wiki page - [with Table of Contents][painters-TOC] and [without][painters-noTOC]. I am interested in these several paragraphs preceding the TOC in the former case, as it may contain a good summary of the painter contribution. In the future I hope to be able to use only this 'abstract' instead of the whole page content for text processing. And if there is no such abstract I just capture the whole content. I achieve this by using [XPath Axes][xpath-axes] and namely the `preceding`and `following` ones. I then use the `lxml.text_content()` to extract text. In the last parsing block I am capturing the infobox with biography if it is available. The infobox is always created as a table, so I parse it as a dictionary with `<th>` values representing keys and corresponding `<td>` values representing values. Finally I store it as a dictionary in a separate json file. 

It takes this function about 20-30 min to download 3148 files. The resulting folder is about 15Mb in size. As I don't expect these pages to be updated frequently there is no need to accelerate it. Otherwise I'd convert it to [Scrapy][scrapy] which supports multiple threads and has many more neat features. I may still do it if/when I decide to parse the connections between painters and extend the dataset, i.e. download the page, extract and follow through all links. Scrapy is definitely more suited for this task.

# Parsing painters

I now have a json file with a master list of artists and the folder with 3,148 files, each containing the text of a wikipedia page broken into abstract, or the section before Table of Contents, the content itself and the infobox with biography details.

Let's extend the master list to create a single json dataset. First I'll create a function to download the content of the file for a given href. It is identical in functionality to the previous one, but just operates locally with already downloaded files. One may populate the DataFrame during download in a same manner using the function above.

{% highlight python %}
#function to get content from a file for a given href
def get_content_from_file(href):
  filename = unicode(href.split('/')[2]).encode(sys.stdout.encoding, 'replace')
  filename = re.sub("\.","",filename)
  with open('painters/'+filename, 'r') as fp:
      content=json.loads(fp.read())
  return content

#download master list into painters
with open(filename, 'r') as fp:
    painters = json.loads(fp.read())
pnt = pd.DataFrame(painters)

#apply and get content
content = pnt.apply(lambda row: get_content_from_file(row['href']), axis=1)
return pd.concat([pnt, pd.DataFrame(list(content))], axis=1, join='inner')
{% endhighlight %}

# Parsing birth and death dates

I am going to use the biography infobox to determine life dates. Again, since it will be used via dataframe.apply() I will be parsing a single row. The logic is simple - I will be looking for 'Born' and 'Died' keys in biography and then extracting the four digit number. If I don't find the 'Born' field in biography dictionary I am going to extract the number from the first sentence.

{% highlight python %}
def parse_content_dates(dfrow):
    result=[None,None]

    #try biography first
    bio = dfrow['biography']
    if bio:
        if 'Born' in bio.keys():
            try:
                result[0] = int(re.findall(u'\d{4}',bio['Born'])[0])
            except IndexError:
                pass
        if 'Died' in bio.keys():
            try:
                death = int(re.findall(u'\d{4}',bio['Died'])[0])
                if death > result[0]:
                    result[1] = death
            except IndexError:
                pass

    #look only at first sentence and only if no Born
    if result[0] is None:
        content = dfrow['sentences'][0]
        try:
            dates = re.findall(u'\d{4}',content)
            if dates:
                result[0] = int(dates[0])
                if len(dates) > 1 and int(dates[1]) > result[0]:
                    result[1]= int(dates[1])
        except IndexError:
            pass
    return result
{% endhighlight %}


The result of the function above looked alright, until I looked at the histogram plot and realized that there are several painters with more than 300 years of life duration:

{% highlight python %}
                      id  bDate  dDate
67           An_Zhengwen   1368   1644
401     Byeon_Sang-byeok   1392   1910
1079              Hu_Zao   1644   1912
2057         Sheng_Maoye   1368   1644
2106         Song_Maojin   1368   1644
2169            Sun_Long   1368   1644
2483             Wu_Hong   1644   1912
2501          Xia_Shuwen   1368   1644
2506             Xie_Sun   1644   1912
2524              Ye_Xin   1644   1912
2536          Yuan_Jiang   1644   1912
2537  Yuan_Yao_(painter)   1644   1912
{% endhighlight %}

It appeared that several Chinese painters with unknown biography have the years of Dynasty instead ([example][chinese-example]). Even the table above has two Dynasty periods: 1368-1644 and 1644-1912. The easy fix is to add a check that life duration should not exceed, say, 120 years to accept the result.

On the same histogram there were about 15 painters with life duration less than 20 or even 10 years. These were also the result of lack of knowledge, e.g. [Pieter de Ring][Pieter_de_Ring] - notice how it says 1615/1620, so the function will treat it as birth and death years and 5 years of life correspondingly. I will add one additional test and if life duration is less than 15 years I will take the next four digit number as the year of death.

After changing it I have 2178 deceased painters and 2514 with at least one date (336 alive) out of 2596 artists recognized as painters.



And since I have this dataset now I can do a simple analysis to see if dealing with beauty on a daily basis extends your life or being a poor genius actually decreases it. First, the plot. I used seaborn library to plot the JointGrid above, the code to generate it is below.

![Life duration]({{site.url}}/assets/arttimeline_painters_life.png)

{% highlight python %}
import seaborn as sns
import statsmodels.api as sm

def plot_dates(painters):
    #data
    dates = painters[['bDate','dDate']].dropna()
    x = dates.bDate
    y = dates.dDate
    
    #plot using seaborn
    sns.set(style="ticks", color_codes=True)
    grid = sns.JointGrid(x="bDate",y="dDate",data=dates,
                         size=10,ratio=10,space=0,xlim={2000,1000})
    fig = grid.plot_joint(sns.regplot)    
    fig = fig.plot_marginals(sns.distplot, kde=True)
    fig.set_axis_labels(xlabel='Birth Year',ylabel='Death Year')

    #add linear regression
    x2fit = sm.add_constant(x)
    model = sm.OLS(y, x2fit).fit()  

    return fig,model
{% endhighlight %}

Unfortunately I didn't find how to extract the regression parameters from the model used in the figure directly and I used the Ordinary Least Squares function from statsmodels module for this purpose. The linear regression fit summary is below:

{% highlight python %}
                           OLS Regression Results                            
==============================================================================
Dep. Variable:                  dDate   R-squared:                       0.990
Model:                            OLS   Adj. R-squared:                  0.990
Method:                 Least Squares   F-statistic:                 2.123e+05
Date:                Wed, 02 Mar 2016   Prob (F-statistic):               0.00
Time:                        13:58:11   Log-Likelihood:                -9358.5
No. Observations:                2211   AIC:                         1.872e+04
Df Residuals:                    2209   BIC:                         1.873e+04
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const         21.5279      3.989      5.397      0.000        13.705    29.350
bDate          1.0262      0.002    460.756      0.000         1.022     1.031
==============================================================================
Omnibus:                      245.671   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              357.041
Skew:                          -0.830   Prob(JB):                     2.95e-78
Kurtosis:                       4.058   Cond. No.                     2.01e+04
==============================================================================
{% endhighlight %}

Judging by the second regression parameter the duration of life is increasing by roughly 2.6 years every 100 years. Also it looks like not much was going on before 1300. Below are painters who died before 1300. Again, majority of them are from China.

{% highlight python %}
                     id  bDate  dDate
950              Guo_Xi   1020   1090
1369              Li_Di   1100   1197
1371         Li_Gonglin   1049   1106
1377  Li_Song_(painter)   1190   1230
1378  Li_Tang_(painter)   1050   1130
1379          Liang_Kai   1140   1210
1438  Ma_Yuan_(painter)   1160   1225
1570              Mi_Fu   1051   1107
2393        Wang_Ximeng   1096   1119
2424           Wen_Tong   1019   1079
2487      Wuzhun_Shifan   1178   1249
2500            Xia_Gui   1195   1224
2529          Yi_Yuanji   1000   1064
2554     Zhang_Shengwen   1163   1189
2562       Zhang_Zeduan   1085   1145
{% endhighlight %}

It makes sense if we recall the Dark Ages in Europe and nothing significant from cultural perspective was going on in Western Culture until Renaissance (ca. 1300). Then there is significant spike around late 1800s. And does not seem that there is any decline trend in new painters (or rather wiki articles about them) in 1990 and early 2000s.

Now, returning to that life expectancy. I'll choose those painters who died in between 2000 and 2010. There are 103 of them. After creating the new column with life duration I ran the statistics and here is the result:

{% highlight python %}
In [193]: recent.life.describe()
Out[193]: 
count    103.000000
mean      81.213592
std       11.382521
min       37.000000
25%       73.500000
50%       83.000000
75%       90.500000
max      101.000000
Name: life, dtype: float64
{% endhighlight %}

The mean of 81 years is higher than the world average life expectancy of 71 years (according to wikipedia), but one has to consider the nationality and country of living to make it a meaningful comparison, so I may do it later after I parse nationalities.

I can't think of any other info I can extract here, so let's move to the next variable.

# Parsing nationality

[story-art]: https://books.google.ca/books?id=1-OfPwAACAAJ&dq
[drawing-right-side]: https://books.google.ca/books?id=2rM8jgEACAAJ&dq
[anaconda]: https://www.continuum.io/downloads
[wiki-painters-list]: https://en.wikipedia.org/wiki/List_of_painters_by_name
[pd-apply]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
[painters-noTOC]: https://en.wikipedia.org/wiki/Leon_Wycz%C3%B3%C5%82kowski
[painters-TOC]: https://en.wikipedia.org/wiki/Mary_Elizabeth_Price
[xpath-axes]: http://www.w3schools.com/xsl/xpath_axes.asp
[scrapy]: http://scrapy.org/
[chinese-example]: https://en.wikipedia.org/wiki/Hu_Zao
[Pieter_de_Ring]: https://en.wikipedia.org/wiki/Pieter_de_Ring