## About/TLDR
As the landscape of data professionals evolves, the recent surge in job opportunities has not been accompanied by a comprehensive analysis of global trends.
To address this gap, our analysis leveraged data from StackOverflow and AI-salaries.net as proxies and, we normalized salaries using Purchasing Power Parity (PPP) to ensure accurate comparisons.

Our findings reveal intriguing insights, particularly highlighting certain countries that exhibit a remarkable susceptibility to changes. 
Notably, Argentina and Russia stand out, experiencing staggering percentage changes throughout the 2019-2023 timeframe we chose.
These fluctuations can be attributed to discernible economic factors, which we go into more detail in [analysis.ipynb](/src/analysis.ipynb).

#### Note about Analysis.ipynb
It can be interactive if you change the country number in the `get_similar_countries(merged, x)` by changing x the analysis of the notebook changes due to a higher number of countries. We chose 20 in this repo since it's much easier to see comparisons without being overwhelmed of say 100 countries.
