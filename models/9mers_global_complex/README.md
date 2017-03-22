## Different models

### 9mers_imbalanced

- batch size: 32


### 9mers_balanced

- batch size: 32

- balanced batches


### 9mers_random

- batch size: 32

- balanced batches

- 20% of the batch is random peptides


### 9mers_weight

- batch size: 32

- balanced batches

- objects weighted according to the exp(B), B - beta distribution


### 9mers_weihe

- batch size: 32

- balanced batches

- objects weighted according to the exp(B), B - beta distribution

- He uniform initialization for all layers