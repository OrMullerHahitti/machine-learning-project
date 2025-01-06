from sklearn.tree import DecisionTreeClassifier, plot_tree ## for training and ploting DT
import data_preparations_part2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
import os


#2.2------------------------------------------------------------------------------------------------------
x_train, y_train, x_test, y_test= data_preparations_part2.prepare_data(data_preparations_part2.data, dummies=True)

TreeModel = DecisionTreeClassifier(random_state=42, criterion='entropy')
TreeModel.fit(x_train, y_train)

print(f"f1_score of train: {f1_score(y_true=y_train, y_pred=TreeModel.predict(x_train)):.2f}")
print(f"f1_score of test: {f1_score(y_true=y_test, y_pred=TreeModel.predict(x_test)):.2f}")

#2.3------------------------------------------------------------------------------------------------------
# Define parameter ranges for max_depth and min_samples_split
max_depth = np.arange(5, 40, 2)  # Reduced range for better performance
min_samples_split = np.arange(10, 710, 50)  # Define a reasonable range for min_samples_split

TreeModel = DecisionTreeClassifier(random_state=42)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
ParamatersDict = {'max_depth': max_depth, 'min_samples_split': min_samples_split}

# Set up GridSearchCV with F1 scoring metric
grid = GridSearchCV(estimator=TreeModel, param_grid=ParamatersDict, scoring='f1', cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
grid.fit(x_train, y_train)
# Extract results into a DataFrame
cv_results = pd.DataFrame(grid.cv_results_)

# Extract only the relevant columns from cv_results_
Selected_max_depth = ['mean_test_score', 'mean_train_score', 'param_max_depth']
DF_Selected_Values = cv_results[Selected_max_depth]
DF_Selected_Values = DF_Selected_Values.groupby('param_max_depth', as_index=False).mean()
# Sort values by max_depth
DF_Selected_Values = DF_Selected_Values.sort_values('param_max_depth', ascending=True)

# Plot for max_depth
plt.figure(figsize=(13, 4))
plt.plot(DF_Selected_Values['param_max_depth'],
         DF_Selected_Values['mean_train_score'], marker='x', markersize=6, label='Train accuracy')
plt.plot(DF_Selected_Values['param_max_depth'],
         DF_Selected_Values['mean_test_score'], marker='o', markersize=6, label='Validation accuracy')
# Add titles and labels
plt.title('Max Depth vs F1 Score')  # Improved title
plt.xlabel('max_depth')  # Clearer axis label
plt.ylabel('F1 Score')  # Clearer axis label
plt.legend()  # Add legend
plt.xticks(DF_Selected_Values['param_max_depth'])  # Show only the tested depths
plt.grid(True)  # Add grid for better readability
plt.show()


# Extract only the relevant columns from cv_results_
Selected_min_samples_split = ['mean_test_score', 'mean_train_score', 'param_min_samples_split']
DF_Selected_Values2 = cv_results[Selected_min_samples_split]

# Group by min_samples_split and calculate the mean
DF_Selected_Values2 = DF_Selected_Values2.groupby('param_min_samples_split', as_index=False).mean()

# Sort values by min_samples_split
DF_Selected_Values2 = DF_Selected_Values2.sort_values('param_min_samples_split', ascending=True)

# Plot for min_samples_split
plt.figure(figsize=(13, 4))
plt.plot(DF_Selected_Values2['param_min_samples_split'],
         DF_Selected_Values2['mean_train_score'], marker='x', markersize=6, label='Train accuracy')
plt.plot(DF_Selected_Values2['param_min_samples_split'],
         DF_Selected_Values2['mean_test_score'], marker='o', markersize=6, label='Validation accuracy')

# Add titles and labels
plt.title('Min Samples Split vs F1 Score')  # Updated title
plt.xlabel('min_samples_split')  # Updated axis label
plt.ylabel('F1 Score')  # Updated axis label
plt.legend()  # Add legend
plt.xticks(DF_Selected_Values2['param_min_samples_split'][::2])  # Show every 2nd value for readability
plt.grid(True)  # Add grid for better readability
plt.show()

#2.4--------------------------------------------------------------------------------------------------------------------------

max_depth_list = np.arange(6, 21)
min_samples_split_list = np.arange(110, 210, 5)

params_dt = {
    'max_depth': max_depth_list,
    'criterion': ['entropy', 'gini'],
    'class_weight': ['balanced', None],
    'min_samples_split': min_samples_split_list,
}
TreeModel = DecisionTreeClassifier(random_state=42)
SKFold_CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# use grid tree to tune the parameters
Grids_Tree = GridSearchCV(estimator=TreeModel, param_grid=params_dt, scoring='f1',
                          cv=SKFold_CV, n_jobs=-1, refit=True, return_train_score=True)
Grids_Tree.fit(x_train, y_train)

best_hyperparameters = Grids_Tree.best_params_
best_TreeModel = Grids_Tree.best_estimator_

Tuned_y_Training_Predictions = best_TreeModel.predict(x_train)
Tuned_Train_f1_score = f1_score(
    y_train, Tuned_y_Training_Predictions)

Tuned_y_Testing_Predictions = best_TreeModel.predict(x_test)

Tuned_Test_f1_score = f1_score(
    y_test, Tuned_y_Testing_Predictions)

print("Best Hyperparameters:", best_hyperparameters)
print('F1 Training score: {:.4f}'.format(Tuned_Train_f1_score))
print('F1 Testing score: {:.4f}'.format(Tuned_Test_f1_score))

# create cv data frame with the results
cv_results = pd.DataFrame(Grids_Tree.cv_results_)
Selected_Leaves = ['mean_test_score', 'mean_train_score', 'std_test_score',
    'param_max_depth', 'param_criterion', 'param_class_weight', 'param_min_samples_split']
DF_Selected_Values = cv_results[Selected_Leaves]
DF_Selected_Values = DF_Selected_Values.sort_values( 'mean_test_score', ascending=False).head(10)
DF_Selected_Values['mean_test_score'] = DF_Selected_Values['mean_test_score'].round(4)
DF_Selected_Values['mean_train_score'] = DF_Selected_Values['mean_train_score'].round(4)
DF_Selected_Values['std_test_score'] = DF_Selected_Values['std_test_score'].round(4)
column_names = {
    'mean_test_score': 'mean test score',
    'mean_train_score': 'mean train score',
    'std_test_score': 'std test score',
    'param_max_depth': 'max depth',
    'param_criterion': 'criterion',
    'param_min_samples_split': 'min samples split',
}

# plot the results
DF_Selected_Values = DF_Selected_Values.rename(columns=column_names)
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
ax.set_title('Grid Search results', y=1.1)
table = ax.table(cellText=DF_Selected_Values.values, colLabels=DF_Selected_Values.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.25, 1.25)
plt.show()

Tuned_y_predicts = best_TreeModel.predict(x_test)

'''**plot tree'''
# Plot the best tree with depth 2
plt.figure(figsize=(22, 15))
plot_tree(best_TreeModel, feature_names=x_train.columns, class_names=True, filled=True, fontsize=6, max_depth=2)
plt.rcParams.update({'font.size': 30})
plt.show()

# Plot the full best tree
plt.figure(figsize=(100, 50))  # Increase figure size for larger trees
plot_tree(best_TreeModel, feature_names=x_train.columns, class_names=True, filled=True, fontsize=6)
plt.title("Decision Tree Visualization", fontsize=16)
plt.savefig("decision_tree_full2.pdf", bbox_inches='tight')  # Save as PDF
print("Decision tree saved to 'optimal_decision_tree.pdf'")
print("Current working directory:", os.getcwd())

# plot the importance features table
importance = pd.DataFrame({
    'Feature_name': x_train.columns,
    'Importance': best_TreeModel.feature_importances_
})
importance = importance.groupby('Feature_name', as_index=False).sum()
importance = importance.sort_values(by='Importance', ascending=False)
importance['Importance'] = importance['Importance'].round(4)
importance = importance.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5, 2))
ax.axis('off')
table = ax.table(cellText=importance.values,
                 colLabels=importance.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(0.5, 0.7)
plt.show()


'''**optimal model'''
optimal_model = DecisionTreeClassifier( min_samples_split=115, max_depth=11, class_weight='balanced', random_state=42)
optimal_model.fit(x_train, y_train)
# get the training and validation set F1 scoring
Tuned_y_Training_Predictions = optimal_model.predict(x_train)
Tuned_Train_F1 = f1_score(y_train, Tuned_y_Training_Predictions)

y_pred_proba = optimal_model.predict(x_test)
Tuned_Test_F1 = f1_score(y_test, y_pred_proba)

print('F1 Training score: {:.4f}'.format(Tuned_Train_F1))
print('F1 Testiing score: {:.4f}'.format(Tuned_Test_F1))
