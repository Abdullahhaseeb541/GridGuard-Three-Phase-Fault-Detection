# Assignment Analysis

This project compares a manually implemented logistic regression model against SVM alternatives for three-phase fault detection. The actual result artifacts are stored under `results/figures/` and `results/tables/`.

## 1. EDA interpretation: current and voltage signatures

- Reference figures: [current_timeseries.png](./figures/current_timeseries.png)
- Your assessed interpretation: [Add your own explanation here. Describe how the current and voltage time-series patterns change during fault events and whether the transitions are clearly separable from normal operation.]

## 2. EDA interpretation: Ia versus Ib separation

- Reference figure: [ia_ib_scatter.png](./figures/ia_ib_scatter.png)
- Your assessed interpretation: [Add your own explanation here. Discuss whether the fault and no-fault samples form distinct clusters and whether the selected features support separation.]

## 3. EDA interpretation: class distribution and feature spread

- Reference figure: [ia_histograms.png](./figures/ia_histograms.png)
- Your assessed interpretation: [Add your own explanation here. Explain what the histogram reveals about the distribution of the current measurements and how it informs model robustness.]

## 4. Logistic regression: learning-rate selection and stability

- Reference table: [table_2_logistic_regression.csv](./tables/table_2_logistic_regression.csv)
- Reference figures: [logistic_cost_history_set_R.png](./figures/logistic_cost_history_set_R.png), [logistic_cost_history_set_Q.png](./figures/logistic_cost_history_set_Q.png)
- Your assessed interpretation: [Add your own explanation here. Explain which learning rate was selected for each set, how the cost traces behaved across iterations, and why the chosen value is appropriate for the assignment.] 

## 5. SVM validation: kernel and regularization trade-offs

- Reference table: [table_3_svm_validation_results.csv](./tables/table_3_svm_validation_results.csv)
- Reference figure: [svm_validation_accuracy_vs_C.png](./figures/svm_validation_accuracy_vs_C.png)
- Your assessed interpretation: [Add your own explanation here. Compare the linear and RBF settings, explain what the validation curve suggests about the effect of C and gamma, and note which configuration is strongest.] 

## 6. Final model comparison and operational interpretation

- Reference table: [table_4_test_results.csv](./tables/table_4_test_results.csv)
- Related files: [table_1_data_preparation.csv](./tables/table_1_data_preparation.csv), [table_5_set_q_weights.csv](./tables/table_5_set_q_weights.csv)
- Your assessed interpretation: [Add your own explanation here. Discuss test-set accuracy, false alarms, missed faults, and the practical meaning of the learned Set Q weights in the context of fault detection.]

## Notes

- The analysis above intentionally leaves the interpretation sections open for your own review and assessment.
- No experimental conclusions are claimed beyond the recorded outputs in the result tables and figures.
