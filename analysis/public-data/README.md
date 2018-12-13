
## Public files for analyzes

This folder contains the questions types and all the information used to obtain results for our [final report](https://github.com/bcsaldias/VQA-analysis/blob/master/WriteUp.pdf). Find in each file the attention types relevance for each question type. Additionally, we provide other features to encourage further exploration.

Note that these public files relate to the abstract scenes and real-world images (MSCOCO) for trainin and validation sets:

	abstract_train_summary.csv
	abstract_vali_summary.csv
	real_train_summary.csv
	real_vali_summary.csv

Each CSV file contains the following columns:

- **\_q\_type**: question type
- **sum_bottom_up**: percentage weight of bottom-up attention by adding the activated features (see section 4.2.2 of the final report)
- **count_bottom_up**: percentage weight of bottom-up attention by counting the activated features (not used in the final report. Note that 50% means that half of the activated features belong to bottom-up features.)
- **_count**: how many question samples are represented by the question type in this question type.
- **winner**: the attention type with biggest percentage weight.
- **my_score**: the absolute number of soft matches reached by our model for the question type in this row.
- **base_score**: the absolute number of soft matches reached by the base model for this question type.
- **above_median**: if the question type contains above median number of samples.
- **o_bottom_up**: total sum of the activated features (before normalizing to obtain sum_bottom_up).
- **o_top_down**: the same as o_bottom_up but for top-down attention.
- **_diff**: sum_bottom_up - sum_top_down = 1 - 2 x sum_bottom_up. To determine whether a question type is responsible for the questions responses, we set a confidence interval (CI) for this value.
- **diff_score**: my_score - base_score
- **percen**: _count / dataset size.

We can also provide the activated features, per each question in each dataset, upon request at belen@mit.edu.
