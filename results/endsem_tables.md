# End term result tables

Generated from 65 recorded end-term runs in `results/baselines.jsonl`, plus `results/evidence.json`.
Regenerate with `code/make_endsem_tables.py`.

Accuracy is quoted at the epoch chosen by a validation set carved out of the
training data at recording-condition level. It is never the best epoch on the
test set, which would be peeking. Table 7 shows what that choice costs.

## Table 1. The accuracy drop, split into its two causes

Every arm of a row uses exactly the same samples and the same test size,
so the only thing that changes across a row is what the split holds back.
The benchmark style arm is a random split over samples, which is what
SenseFi ships. The grouped arm is also random and still contains every
volunteer, position and orientation, but no repeat take of one recording
condition is allowed to straddle it. The cross arm holds out a whole
condition.

| Model | Protocol | Benchmark style | Grouped in domain | Cross domain | Cost of removing repeat takes | Cost of the domain shift | Total |
|---|---|---|---|---|---|---|---|
| LeNet | Held out orientation | 0.687 | 0.638 | 0.326 | -0.049 | -0.313 | -0.361 |
| LeNet | Held out position | 0.681 | 0.623 | 0.702 | -0.059 | +0.080 | +0.021 |
| LeNet | Held out volunteers | 0.748 | 0.736 | 0.594 | -0.012 | -0.142 | -0.154 |
| GRU | Held out orientation | 0.627 | 0.567 | 0.267 | -0.061 | -0.300 | -0.361 |
| GRU | Held out position | 0.619 | 0.573 | 0.625 | -0.046 | +0.052 | +0.006 |
| GRU | Held out volunteers | 0.698 | 0.696 | 0.576 | -0.002 | -0.119 | -0.121 |
| ViT | Held out orientation | 0.660 | 0.594 | 0.321 | -0.066 | -0.273 | -0.339 |
| ViT | Held out position | 0.669 | 0.610 | 0.682 | -0.059 | +0.072 | +0.013 |
| ViT | Held out volunteers | 0.700 | 0.691 | 0.535 | -0.009 | -0.155 | -0.165 |
| ResNet18 | Held out orientation | 0.689 | 0.617 | 0.267 | -0.071 | -0.351 | -0.422 |
| ResNet18 | Held out position | 0.683 | 0.634 | 0.689 | -0.049 | +0.056 | +0.006 |
| ResNet18 | Held out volunteers | 0.715 | 0.683 | 0.558 | -0.031 | -0.126 | -0.157 |

## Table 2. Leave one condition out, every value in turn

LeNet, cross arm only. If the drop were a property of the dataset as a
whole the row would be flat. It is not.

| Orientation held out | 1 | 2 | 3 | 4 | 5 | Spread |
|---|---|---|---|---|---|---|
| Accuracy | 0.450 | 0.611 | 0.596 | 0.458 | 0.326 | 0.285 |

| Position held out | 1 | 2 | 3 | 4 | 5 | Spread |
|---|---|---|---|---|---|---|
| Accuracy | 0.658 | 0.600 | 0.653 | 0.608 | 0.702 | 0.102 |

## Table 3. How much of the loss each recovery method wins back

LeNet on the cross arm. The first row is the same model with no recovery
applied, and every later row is the change against it. A negative number
means the method made things worse, which is reported as readily as a
positive one.

| Method | Held out orientation | Held out position | Held out volunteers |
|---|---|---|---|
| No recovery | 0.326 | 0.702 | 0.594 |
| augment 20 deg | 0.321 (-0.005) | 0.731 (+0.029) | 0.614 (+0.020) |
| augment 20 deg + mixup | 0.350 (+0.025) | 0.742 (+0.040) | 0.631 (+0.037) |
| augment 45 deg | 0.370 (+0.044) | 0.716 (+0.013) | 0.604 (+0.010) |
| mixup | 0.312 (-0.013) | 0.703 (+0.001) | 0.581 (-0.013) |

## Table 4. Does the recovery cost anything, and does it transfer

A method that raises cross domain accuracy is only worth having if it does not
quietly cost in domain accuracy to get there. The first rows apply the same
augmentation to the grouped in domain arm. The last row repeats it on a different
architecture.

| Model | Protocol | Arm | Method | Without | With | Change |
|---|---|---|---|---|---|---|
| LeNet | Held out orientation | Grouped in domain | augment 20 deg | 0.638 | 0.655 | +0.016 |
| LeNet | Held out position | Grouped in domain | augment 20 deg | 0.623 | 0.664 | +0.041 |
| LeNet | Held out volunteers | Grouped in domain | augment 20 deg | 0.736 | 0.730 | -0.006 |
| ResNet18 | Held out orientation | Cross domain | augment 20 deg | 0.267 | 0.304 | +0.038 |

## Table 5. The same split answered by lookup, with no model

Each test sample is given the gesture of the single most similar training
sample, measured by cosine similarity on the raw data. No training and no
parameters are involved, so nothing here can be blamed on an architecture
or a learning rate.

| Protocol | Arm | Classes | Chance | Nearest neighbour accuracy |
|---|---|---|---|---|
| Held out orientation | Benchmark style | 22 | 0.045 | 0.306 |
| Held out orientation | Grouped in domain | 22 | 0.045 | 0.231 |
| Held out orientation | Cross domain | 22 | 0.045 | 0.101 |
| Held out position | Benchmark style | 22 | 0.045 | 0.313 |
| Held out position | Grouped in domain | 22 | 0.045 | 0.233 |
| Held out position | Cross domain | 22 | 0.045 | 0.265 |
| Held out volunteers | Benchmark style | 9 | 0.111 | 0.381 |
| Held out volunteers | Grouped in domain | 9 | 0.111 | 0.328 |
| Held out volunteers | Cross domain | 9 | 0.111 | 0.250 |

## Table 6. What makes two recordings look alike

| The pair shares | Pairs | Mean cosine similarity |
|---|---|---|
| Same recording condition | 1500 | 0.565 |
| Same gesture, different condition | 1500 | 0.449 |
| Different gesture | 1385 | 0.437 |

## Table 7. What the choice of reporting epoch is worth

Three ways of reading the same runs. The final epoch is what SenseFi's
own script reports, since it trains for a fixed number of epochs and
tests once at the end. The validation selected column is what this report
quotes. The oracle column is the best epoch measured on the test set,
which is not a number anyone could obtain in practice and is shown only
to size the temptation.

| Reading | Mean accuracy over 36 cross-domain and in-domain runs |
|---|---|
| Final epoch, as SenseFi reports | 0.587 |
| Validation selected, as this report quotes | 0.611 |
| Best test epoch, an oracle | 0.619 |

## Table 8. The benchmark's CNN+GRU applies softmax twice

`Widar_CNN_GRU` ends in `nn.Softmax`, and the training loop then applies
`nn.CrossEntropyLoss`, which softmaxes again. The same mistake is present
in the UT-HAR and NTU-Fi versions of that model. Removing the layer is a
one line change and it is worth this much accuracy.

| Version | Accuracy |
|---|---|
| As shipped, softmax applied twice | 0.652 |
| Trailing softmax removed | 0.759 |

