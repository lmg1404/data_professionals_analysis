## ReadMe
The rapidly evolving data professionals labor market presents numerous opportunities to elucidate insights for emerging professionals.
In taking advantage of these opportunities, our analysis leveraged data from StackOverflow Developer surveys and AI-salaries.net survey as a plausibly representative sample of data professionals and created a standardized measure of compensation values across these surveys, across survey years, and across geographies with OECD Purchasing Power Parity (PPP) factors.


The standardized compensation metric (the 2023 USD) allows for comparisons and analytics that serve as a window into global data professionals compensation trends unavailable with the raw data.


Notable examples of available insights being the dramatic swings of Argentina and Russia data professionals` compensations throughout the 2019-2023 timeframe we studied.
These fluctuations correlate with commensurate economic shocks, which are discussed in [analysis.ipynb](/src/analysis.ipynb).


#### Building upon Analysis.ipynb
Quick example of how one might expand on the analysis done here — change the country number in the `get_similar_countries(merged, x)` by adjusting the value of x to create a visualization comparing the distributions of compensations for x number of countries.
