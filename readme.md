## Total exchange flow analysis framework for regional ocean modeling system outputs.

'tef' contains scripts used to implement the total exchange flow analysis framework for the coastal ocean. TEF was developed by Parker MacCready for analyzing estuarine exchange flow and mixing (e.g. MacCready, 2011 & MacCready et al., 2018).

The two scripts below isolate a control volume specified by the user and bin tracer transport in salinity coordinates, then compute TEF:

	histograms.py 
	exhange_flow.py
To look at the time rate of change of the control volume size and volume integrated salinity, salinity squared, and salinity variance, run:

	tendency_budgets.py
To calculate salinity variance dissipation, run:
	
	chi.py
The outputs of these scripts are stored in the outputs folder in several subdirectories. 

### Recent updates: 
I've started looking at larger control volumes that are several hundred square kilometers, which is in the largecv folder. Even with dask, the requested memory is too large and therefore saving to a single netcdf is difficult. The best solution I've found is to subset the data and load in the histogram points each day and save the histograms then. Even though this method is annoying, you can easily automate merging and cleaning directories with some simple bash scripts.

To make the code more efficient, I placed the common TEF functions in:

    functions.py
I've also started looking at different normalization schemes too and other ways to bin the data. For example, binning in salinity and water parcel depth to gain some insight into the dynamics. Eventually, I'll merge this into functions.py. Coming soon - adding the integral relationships regarding entrainment and diffusive salt flux from Wang et al. (2017) JPO. The prelimary results of that can be seen in largecv/Budgets_2010.ipynbß