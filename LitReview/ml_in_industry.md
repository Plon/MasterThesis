# Machine Learning in Industry: Why It's Important. 
In 2017, The Economist famously declared that data had overtaken oil to become the world's most valuable resource \cite{theeconomist}. 
However, just like oil, raw data has little value in itself. 
Only through refinement does it become valuable. 
Only when data is gathered and analyzed does it become valuable information. 
There is no shortage of data. 
64.2 zettabytes of data were created in 2020 \cite{stat9sta}. 
The Statista Research Department estimates that in the coming years the amount of c
data created will grow at a compound annual rate of 19.2 % \cite{stat9sta}. 
Consequently, in 2025 there will be three times as much data generated as just five years ago. 
This enormous amount of data presents a massive opportunity for those who can extract valuable information from it. 
Doing that, however, is challenging. 
Obtaining valuable insights, predictions, and decisions from this ocean of data can be compared to finding a needle in a haystack. 


Data availability coupled with low-cost computing has propelled machine learning from a niche research area to a widespread practical and commercial application. 
Accessible cloud computing, powerful pre-trained models, and open-source libraries have made it simple to translate ML into commercial applications. 
This impact extends beyond tech companies. 
Businesses that deal with data-intensive issues have realized the value of ML. 
A 2017 MIT Technology Review survey\cite{mittechreview} of 375 leading businesses in over 30 countries found that 60 % of respondents had already implemented ML strategies and had committed to ongoing investments. 
The ability to gain a competitive advantage was the most important benefit of ML, according to respondents. 
26 % of respondents believed they had already achieved this goal. 
ML is incorporated into businesses' core processes for a variety of strategic reasons. 



## Revenue and Growth
Data analytics are used in business to identify patterns and make strategic decisions that drive revenue growth. 
Machine learning can increase the volume, speed, and quality of the insights of existing data analytics infrastructure. 
Unlike conventional statistical models, ML incorporates previous performance into the model and can adapt to different market environments. 
The value ML adds to a business is reflected in the income statement. 
A study of 330 public North American companies found that companies using data analytics in their decision-making were on average 6 % more profitable than their competitors \cite{hbr2012}. The performance difference remained even after accounting for the investments into ML projects. 



## Time and Efficiencies 
In most industries, labor costs represent the single largest expense. 
Labor efficiencies are therefore a major argument for implementing ML in an organization. 
The average product of labor is a concave function. 
Initially, specialization and division of labor increase productivity. 
However, at some point adding more employees reduces the average product of labor. 
At that point adding more units of labor decreases average productivity. 
Unless the average wage also decreases by at least the same factor, scaling operations by adding more employees will not be an economically viable option. 
This is a well-known problem in scaling operations where labor is a significant factor of production. 
The solution to this problem can be found in machine learning. 
Using ML models, scaling the output can be as simple as modifying a single variable. 
As defined by Tom Mitchell \cite{mitchell1997machine}, machine learning algorithms improve their performance through experience. 
The extra output will improve the model's performance, making it even more effective. 
As a result, ML's productivity scales better than linear. 



## Investment Costs
ML projects require a significant investment in obtaining and storing data, human resources, and implementation. 
Data quality is a critical component of developing and training ML models, and procuring it can be expensive upfront. 
Each stage of an ML project requires businesses to determine how to utilize their human resources effectively. 
In many projects, third-party platforms are engaged at some point. 
Those who are unable to invest in data science capabilities might consider hiring a third-party vendor to develop their customized ML project. 
Developing a successful project, including data collection, development and training of a model, and implementation, takes 12 months on average \cite{deloitte}. 
The length of time will vary according to the complexity of the problem. 














# Machine Learning in Academia vs. Industry.
The recent success of Machine Learning has attracted a great deal of interest from academia and industry. Both have contributed to shaping modern ML. 
Academic and industrial ML is certainly coextensive. 
Academia does not only conduct pure and basic research. 
Industry does not only conduct applied and ad hoc research. 
There are no universal differences between machine learning in academia and industry. 
Nevertheless, there are some characteristic differences between academic and industrial ML. 




## Purpose.
There is pressure on academics to regularly get their work published, as expressed by the aphorism “Publish or Perish”. 
There is a strong emphasis on novelty in academia, and on contributing some new knowledge to the field. 
If you make some theoretical breakthroughs, then you will have a better chance of getting published in respected academic journals. 
As a result, academics may err on the side of theoretical complexity. 
The research will also be narrow in scope. 
The methodology is a key aspect of academic research, as it is scientific work and reproducibility is vital. 

Research in an industry context places more emphasis on aiding business objectives. 
Academic research is primarily judged by peers, whereas research in industry is primarily judged by laymen. 
To convince management to invest in a machine learning project, the researchers need to present immediate tangible business benefits. 
After all, management is result oriented and sees ML merely as a tool for achieving them. 
ML is used to deliver business impact. Industry will invest in research that advances that goal. 
Scientific truth or the novelty of a model has in itself no value in this context.
In fact, simplicity is often preferred as it is more cost-effective. 





## Data
In academia, ML models are typically trained on small clean datasets. The datasets have been curated specifically for that purpose. 
Unless the research is specifically about feature engineering, it is unlikely to receive much attention. 
However, this is rarely how it works in an industry setting. 
Data is messy, values are often missing, there might be duplicate observations, etc. 
Cleaning up datasets is a necessary step before training models. 
To train a model effectively, feature engineering is essential for most industrial projects. 
Companies may have access to proprietary data that is not available to the public. 




## Speed
In industry speed to market is key. 
Agile development is often utilized to quickly deploy ideas to production, and continually improve. 
Industry must constantly adapt to competition and changing markets. 
Management must allocate the company's resources effectively. 
A marginal improvement in a machine learning model's performance might not be worth the additional investment. 
Management might prefer a quicker-to-develop model over a more time-consuming project with superior performance unless the additional performance benefits are critical. 

In academia, accuracy might take precedence over speed. 
Both time and budget constraints still apply. 
However, academics will strive for perfection within those constraints. 
Academics can spend years researching a model that initially shows little or no commercial value. 
Through this type of research, new discoveries can be made that push the field forward. 



## Infrastructure. 
ML at an industrial scale has additional requirements compared to academia. 
It is perfectly feasible to do ML research in academia using a .csv file in a Jupyter Notebook environment on a PC. 
Academic models usually aren't built with scalability and long-term goals in mind.
For industrial ML, establishing a robust ML infrastructure is essential. 
Databases, models, and performance all need to be managed effectively. 
Industry uses end-to-end ML pipelines to automate the workflow of ML projects. 
It encompasses everything from data extraction and preprocessing to training and deployment. 

Performance monitoring is crucial for ML systems. 
An alert system might be useful if performance drops below a certain threshold.
Models with different levels of complexity are commonly run on the same problem. 
If their performance deviates from expectations for extended periods of time, then something may not be functioning properly. 

ML projects can be resource-intensive, especially when using large datasets. 
Long-term industrial projects require resource efficiency. 
In-house development and operation of this infrastructure are expensive. 
Third-party vendor cloud-based platforms like Microsoft Azure, Amazon Web Services, and IBM Cloud are commonly used.
They offer Machine Learning as a Service (MLaaS) that makes ML more accessible. 
Platforms like these provide industry with the tools to easily scale ML projects while at the same time reducing costs due to only paying for the resources that are being used. 
