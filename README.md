# Monetary Policy Coordination

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15067958.svg)](https://doi.org/10.5281/zenodo.15067958)

Replication code for the manuscript *Measuring Monetary Policy Coordination from Central Bankers' Speeches*.

## Data

The datasets used in this work were:
- [Central bankers' speeches](https://www.bis.org/cbspeeches/index.htm)
- [Central bank policy rates](https://data.bis.org/topics/CBPOL)
- [Central bank total assets](https://data.bis.org/topics/CBTA)

For speech data collection, we used the code made available by Hansson (2021) at https://github.com/HanssonMagnus/scrape_bis. This script populates `1_raw_txt`.


## Code

The files in this repository are:
### monetary_policy_coordination.py
Speech similarity network code.

### Data Preparation.ipynb
Jupyter notebook used to find files with encoding errors in `1_raw_txt` and send to `1_raw_txt_error`.

### Monetary Policy Coordination.ipynb
This is the main jupyter notebook used in the work.

It imports `monetary_policy_coordination.py` to build and analyse the speech similarity network.

The first part, **Data Processing**, is used to generate the datasets used in the study. It takes files from `1_raw_txt` and builds the files `Central_Bank_Index.csv`, `Term_Document.csv` and `Word_Index.csv` in `Processed_Data`.

Then it performs following analyses, presented in the manuscript:
- Long-term speech similarity;
- Evolution of Monetary Policy Coordination;
- Word-level analyses.

### BIS Conventional Data.ipynb
This jupyter notebook is used for the benchmark monetary policy coordination networks, built from policy rates and QE/QT.

It first imports the data and saves the files `WS_CBPOL.csv` and `WS_CBTA.csv` to `Conventional_Coordination_Data`.

Then it performs the analyses on such data, including building coordination networks.


## References

Bank for International Settlements (2024), *Central bank policy rates, BIS WS_CBPOL 1.0 (data set)*, https://data.bis.org/topics/CBPOL/data (accessed on 05 September 2024).

Bank for International Settlements (2024), *Central bank total assets, BIS WS_CBTA 1.0 (data set)*, https://data.bis.org/topics/CBTA/data (accessed on 05 September 2024).

Bank for International Settlements (2024). *Central bank speeches, 1997-2024*, https://www.bis.org/cbspeeches/download.htm.

Hansson, M. (2021). *Evolution of topics in central bank speech communication*. Retrieved from University of Gothenburg, Department of Economics website: https://EconPapers.repec.org/RePEc:hhs:gunwpe:0811
