## Overall Insights

The exploratory data analysis provides a comprehensive understanding of the borrower population and the factors associated with loan default.

- The dataset is **imbalanced**, with non-default loans substantially outnumbering default loans. Therefore, model evaluation should rely on metrics such as Precision, Recall, F1-score, and ROC-AUC rather than accuracy alone.

- Most borrowers are relatively **young adults** with annual incomes below **$125,000**, employment histories shorter than **10 years**, and credit histories of less than **10 years**. The majority of loans are relatively small, while most borrowers receive high loan grades (Grades A and B).

- The typical borrower **does not fully own a home**, most commonly rents or has a mortgage, applies for loans primarily for **education**, and has **no previous default history**.

- Financial characteristics exhibit stronger relationships with loan default than demographic characteristics. Borrowers with **higher loan-to-income ratios**, **lower annual incomes**, **larger loan amounts**, **higher interest rates**, and **lower loan grades** consistently show higher default risk. In contrast, age and credit history length demonstrate only weak relationships with loan performance.

- Previous repayment behavior is one of the strongest categorical indicators of credit risk. Borrowers with prior default records are approximately twice as likely to default compared with those without previous defaults.

- Correlation analysis confirms that **loan grade** and **loan-to-income ratio** are the variables most strongly associated with the target variable. Moderate relationships are also observed for **interest rate** and **annual income**, while age and credit history length contribute relatively little predictive information.

- Feature relationship analysis further explains these patterns. Higher-income borrowers generally receive larger loans but allocate a smaller proportion of their income to debt obligations. Better loan grades are consistently associated with lower interest rates, while older borrowers naturally possess longer employment histories and credit histories.

### Key Findings

Overall, the analysis indicates that **borrowers' financial capacity and historical credit behavior are considerably more important than demographic characteristics when assessing default risk**. Among all available variables, **loan grade**, **loan-to-income ratio**, **previous default history**, **interest rate**, and **annual income** emerge as the most informative predictors of loan default. These findings provide a solid foundation for feature selection and the development of an effective Credit Risk Scorecard model.