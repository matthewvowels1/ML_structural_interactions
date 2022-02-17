# ML_structural_interactions
This notebook demonstrates the interactions between machine learning models and data-generating structures.

The purpose of the notebook is to demonstrate that even with flexible, data-adaptive machine learning techniques, it is non-trival to avoid falling prey to the interactions between the (potentially unknown) structure of the data generating process.

For instance, even if two variables A and B are highly statistically associated with a third variable Y, a random forest will assign variable A an importance close to 0 if it is succeeded by B as a mediator between A and Y.

This makes importances and / or Shapley values impossible to interpret reliably without a strong understanding of the underlying structure.
