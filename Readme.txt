Dataset format
- name_views.mat - Attributes views in n_samples*n_features format
- name_links.mat - Relational views in adjacency matrix format
- truth.mat - Label matrix (0-1 encoding)
- raw_ids.mat - 1-nsamples vector
- labelled_indices_perc_*/**.mat - fold wise indexes
  	* - labeled data ratio   ** - fold 
	Indexes should be created based on stratified snowball sampling

---------------------------------------
Results are printed to *_Data/results_** filenames
	* - Dataset name ** - labeled data ratio

Result file format
- Colums in results_* correspond to Accuracy Precision Recall F1 Hamming-Accuracy Exact-Accuracy Label-Accuracy
- Rows correspond to fold wise results where the last row gives the average results

----------------------------------------
Main file 
- CoTraining.m
