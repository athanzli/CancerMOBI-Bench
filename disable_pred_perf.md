# Disabled Predictive Performance Prints

All sample-level predictive performance metric prints (accuracy, AUC-ROC, AUCPR, F1, precision, recall, MCC, balanced accuracy) have been commented out across the codebase to avoid showing them to users in the public GitHub version. Only biomarker identification output and running time prints are kept.

**To restore**: uncomment the lines listed below (remove the leading `# `).

---

## code/models.py

- **Line 762**: `print(f'Validation set performance: {perf}')` — CustOmics validation perf
- **Line 950**: `print(f'Validation set performance: {perf}')` — TMONet validation perf
- **Line 991**: `print("Test set performance:", perf)` — TMONet test perf
- **Lines 1393–1398**: AUC-ROC, AUCPR, F1, Precision, Recall, MCC (Stabl block)
- **Lines 1402–1403**: F1-weighted, F1-macro (Stabl block)
- **Line 1406**: Accuracy (Stabl block)
- **Line 1419**: `print("Performance:", perf)` (Stabl block)
- **Lines 1581–1586**: AUC-ROC, AUCPR, F1, Precision, Recall, MCC (SVM block)
- **Lines 1590–1591**: F1-weighted, F1-macro (SVM block)
- **Line 1593**: Accuracy (SVM block)
- **Line 1606**: `print("Performance:", perf)` (SVM block)
- **Lines 1671–1676**: AUC-ROC, AUCPR, F1, Precision, Recall, MCC (RF block)
- **Lines 1680–1681**: F1-weighted, F1-macro (RF block)
- **Line 1683**: Accuracy (RF block)
- **Line 1696**: `print("Performance:", perf)` (RF block)

## code/selected_models/DIABLO/run_diablo.py

- **Lines 207–208**: AUC-ROC, AUCPR
- **Lines 209–211**: F1, Precision, Recall
- **Line 212**: MCC
- **Line 214**: Accuracy
- **Line 229**: `print("Performance:", perf)`

## code/selected_models/DeePathNet/scripts/run_deeppathnet_forbk.py

- **Lines 427–428**: AUC-ROC, AUCPR
- **Lines 432–434**: F1, Precision, Recall
- **Line 436**: MCC
- **Line 438**: Balanced Accuracy
- **Lines 442–443**: F1-weighted, F1-macro
- **Line 445**: Accuracy

## code/selected_models/DeePathNet/scripts/models.py

- **Line 457**: `print(f"Epoch {epoch} validation f1:{f1:4f}, val loss:{val_loss:4f}")` — training loop epoch print

## code/selected_models/DeepKEGG/run_deepkegg_forbk_pytorch.py

- **Line 623**: Epoch training print (Train Loss, Val Loss, val_accuracy)
- **Lines 666–667**: AUC-ROC, AUCPR
- **Lines 671–673**: F1, Precision, Recall
- **Line 675**: MCC
- **Lines 681–682**: F1-weighted, F1-macro
- **Line 684**: Accuracy

## code/selected_models/MOGONET/train_test.py

- **Lines 268–270**: Test ACC, Test F1, Test AUC (binary branch in training loop)
- **Lines 273–275**: Test ACC, Test F1 weighted, Test F1 macro (multiclass branch in training loop)
- **Lines 348–352**: AUC-ROC, AUCPR, F1, Precision, Recall
- **Lines 356–357**: F1-weighted, F1-macro
- **Line 359**: Accuracy

## code/selected_models/MOGLAM/train_test.py

- **Lines 250–252**: Val ACC, Val F1 weighted, Val F1 macro (training loop)
- **Lines 319–320**: AUC-ROC, AUCPR
- **Lines 324–326**: F1, Precision, Recall
- **Line 328**: MCC
- **Lines 334–335**: F1-weighted, F1-macro
- **Line 337**: Accuracy

## code/selected_models/MORE/Code/train_test.py

- **Lines 261–263**: Test ACC, Test F1, Test AUC (binary branch in training loop)
- **Lines 266–268**: Test ACC, Test F1 weighted, Test F1 macro (multiclass branch in training loop)
- **Lines 338–342**: AUC-ROC, AUCPR, F1, Precision, Recall
- **Lines 346–347**: F1-weighted, F1-macro
- **Line 349**: Accuracy

## code/selected_models/MoAGLSA/train_test.py

- **Lines 282–285**: Epoch Val ACC, F1 weighted, F1 macro, Train/Val Loss (training loop)
- **Lines 350–351**: AUC-ROC, AUCPR
- **Lines 355–357**: F1, Precision, Recall
- **Line 359**: MCC
- **Lines 365–366**: F1-weighted, F1-macro
- **Line 368**: Accuracy

## code/selected_models/GNNSubNet/GNNSubNet/GNNSubNet.py

- **Lines 432–433**: AUC-ROC, AUCPR
- **Lines 437–439**: F1, Precision, Recall
- **Line 441**: MCC
- **Line 443**: Balanced accuracy
- **Lines 447–448**: F1-weighted, F1-macro
- **Line 450**: Accuracy
- **Line 608**: `print("Accuracy: {}%".format(accuracy))` (eval loop)

## code/selected_models/GNNSubNet/GNNSubNet/OMICS_workflow.py

- **Line 268**: `print("Accuracy: {}%".format(accuracy))`

## code/selected_models/GNNSubNet/GNNSubNet/gnnexplainer_sim.py

- **Line 114**: `print("Accuracy: {}%".format(accuracy))`

## code/selected_models/GNNSubNet/GNNSubNet/gnnexplainer_vanilla_sim.py

- **Line 114**: `print("Accuracy: {}%".format(accuracy))`

## code/selected_models/GENIUS/Training/run_genius_for_bk.py

- **Line 248**: Epoch Val Loss, Val Accuracy (training loop)
- **Lines 289–290**: AUC-ROC, AUCPR
- **Lines 294–296**: F1, Precision, Recall
- **Line 299**: MCC
- **Lines 304–305**: F1-weighted, F1-macro
- **Line 307**: Accuracy

## code/selected_models/PNet/run_pnet.py

- **Line 264**: Epoch training print (Train Loss, Val Loss, Val Acc, lr)
- **Lines 338–339**: AUC-ROC, AUCPR
- **Lines 340–343**: F1, Precision, Recall, MCC
- **Lines 347–348**: F1-weighted, F1-macro
- **Line 350**: Accuracy

## code/selected_models/CustOmics/src/network/customics.py

- **Lines 584–585**: AUC-ROC, AUCPR
- **Lines 589–591**: F1, Precision, Recall
- **Line 593**: MCC
- **Lines 599–600**: F1-weighted, F1-macro
- **Line 602**: Accuracy

## code/selected_models/Pathformer/Pathformer_code/run_pathformer_for_evalbk.py

- **Lines 446–449**: ACC_train, auc_train, f1_weighted_train, f1_macro_train (binary training loop)
- **Lines 479–482**: ACC_val, auc_val, f1_weighted_val, f1_macro_val (binary validation loop)
- **Lines 540–545**: AUC-ROC, AUCPR, F1, Precision, Recall, MCC (binary final eval)
- **Lines 603–609**: acc_train, auc_weighted_ovr/ovo_train, auc_macro_ovr/ovo_train, f1_weighted_train, f1_macro_train (multiclass training loop)
- **Lines 642–648**: acc_val, auc_weighted_ovr/ovo_val, auc_macro_ovr/ovo_val, f1_weighted_val, f1_macro_val (multiclass validation loop)
- **Lines 693–694**: F1-weighted, F1-macro (multiclass final eval)
- **Line 696**: Accuracy (multiclass final eval)

## code/selected_models/TMONet/train/forbk_train_tmonet.py

- **Lines 487–492**: AUC-ROC, AUCPR, F1, Precision, Recall, MCC
- **Lines 496–497**: F1-weighted, F1-macro
- **Line 499**: Accuracy

## code/selected_models/TMONet/train/train_tcga_pancancer_multitask.py

- **Lines 415–416**: Train loss, Acc, Precision, Recall, F1 (training loop)
- **Lines 462–463**: Test loss, Acc, Precision, Recall, F1 (training loop)
