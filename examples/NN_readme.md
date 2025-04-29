# Description of each NN

- NN1:
  - dt=1ns
  - no eq points
  - TP not more weighted
- NN3:
  - dt=1ns
  - eq points
  - TP more weighted
- NN5:
  - dt=1ns
  - eq points
  - TP more weighted
  - IDT+ZND instead of previous PSR+ZND
- NN6:
  - NN5 with last hidden layer preserving conservation laws

- NN13:
  - dt=1ns
  - 550k data: 0D CV + ZND
  - no extra eq points
  - all weighted the same
  - TPY -> dT, dP, dY, not Boxcox transformed
  - input and output are normalized: data=(data-mean(data))/std(data)
- NN16:
  - same training data as NN13
  - TPC -> dT, dP, dC-S*dt
    - C is the concentration, S is the net production rate of each species.
    - C_new = C_old + S*dt + O(dt^2), NN is predicting O(dt^2)
