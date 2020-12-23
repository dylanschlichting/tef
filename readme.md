## Total exchange flow analysis framework for regional ocean modeling system outputs

'tef' contains scripts used to implement the total exchange flow analysis framework for the coastal ocean. TEF was developed by Parker MacCready for analyzing estuarine exchange flow and mixing (e.g. MacCready, 2011 & MacCready et al., 2018). The twoscripts below isolate a control volume specified by the user and bin tracer transport in salinity coordinates, then compute TEF:

	histograms.py 
	exhange_flow.py
To look at the time rate of change of the control volume size and volume integrated salinity, salinity squared, and salinity variance, run:

	tendency_budgets.py
To calculate salinity variance dissipation, run:
	
	chi.py
The outputs of these scripts are stored in the outputs folder in several subdirectories. In the future, I plan to further functionalize this repo and turn TEF into a package. I also plan to add in provisions for using other coordinate systems such astemperature and increase the amount of available temporal averaging tools.
