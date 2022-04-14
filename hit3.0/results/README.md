# Results scripts and tools 

Several scripts for munging various results files 

- `process_csv.py`: for computing inter-annotator agreement on a csv results file. Can handle either MTurk or Turkle results files 
- `process_for_mturk.py`: similar to `process_csv.py` but for pre-processing data for running the MTurk pilot. Extracts the examples for which the top 2 annotators in the Turkle pilot agreed at the Skip level, turns those into a file to upload to MTurk to run the pilot.  
- `extract_good_mturk.py`: extract the annotations for good annotators from the mturk pilot. Reads in the mturk csv and a line-separated file of annotator ids, writes the annotations belongign to those annotators to another file. 
- `merge_mturk_turkle.py`: takes an mturk results file and a turkle results file, merges them into a single turkle-format results file for processing, so that we can compute the agreement between mturk and turkle annotatators 


## Pipeline 
1. ran Turkle pilot for 1 hours with 2 annotators and myself 
    - results: `csvs/pilot_data_from_sorted_by_difference_full-Batch_1014_results.csv` 
2. computed IA on pilot data to get best of 2 annotators (me, ohussei3) 
3. used `process_for_mturk.py` to extract csv of examples that we both did and agreed on, subsampled to have a 2:1 ratio of skips to non-skips 
4. ran mturk pilot with 41 examples from our pilot HIT 
    - results: `mturk/Batch_4710906_batch_results.csv` 
5. used `process_csv.py` with `--mturk` flag to get annotators who completed sufficient number of examples, put those into `mturk/pilot_good_anns.txt` 
6. use `extract_good_mturk.py` to get just their annotations from the larger MTurk csv 
    - results: `mturk/pilot_good_anns.csv` 
7. use `merge_mturk_turkle.py` to merge the annotations with the turkle pilot data
    - results: `mturk/good_with_turkle_merged.csv` 
8. use `process_csv.py` to process the merged file and get IA between turkle and mturk annotators 
