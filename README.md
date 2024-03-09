## About/TLDR
The rapidly evolving data professionals labor market creates numerous opportunities to elucidate valuable insights for emerging professionals.
To address this, our analysis leveraged data from StackOverflow and AI-salaries.net as a plausibly representative sample of data professionals and created a standardized measure of compensation values across these surveys, across survey years, and across geographies with OECD Purchasing Power Parity (PPP) factors.


The standardized compensation metric, the 2023 USD, allows for comparisons and analytics that may serve as a window into global data professional compensation trends unavailable with the raw data.


Briefly, notable examples being dramatic swings of Argentina and Russia data professionals` compensations throughout the 2019-2023 timeframe we chose.
These fluctuations correlate with commensurate economic shocks, which we go into more detail in [analysis.ipynb](/src/analysis.ipynb).


#### Note about Analysis.ipynb
Quick example of expanding on the analysis done here — change the country number in the `get_similar_countries(merged, x)` by changing increasing the value of x to compare the distributions of compensations for x number of countries.

