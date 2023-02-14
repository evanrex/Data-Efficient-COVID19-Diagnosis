# DATA-EFFICIENT METHODS IN MEDICAL IMAGE ANALYSIS OF COVID19 CHEST CT SCANS.
Honours Research Project


An open question for certain tasks in the Medical AI and Computer Vision space is
whether pre-training on domain-specific data yields a statistically significant improve-
ment in performance as opposed to pre-training on non-domain-specific data. One such
task where this question is yet unanswered is that of diagnosing COVID19 from chest
Computed Tomography (CT) scans. A further question is what pre-training methods
might be effective in bringing about this improvement. In this paper, we show that for
the downstream task of COVID19 classification, pre-training a ResNet50 model with
the self-supervised learning technique of DINO on an unlabelled domain-specific data
set of fewer than 100 000 CT images yields a statistically significant improvement over
using a model with pre-trained ImageNet weights. The Area Under the Receiver Op-
erating Characteristic (ROC) Curve (AUC) is used to quantify this performance, while
Permutation Testing is used to show that the increase is statistically significant.

# Reproducibility

We use slurm to run our exoeriments, see this [sbatch file](https://github.com/evanrex/Data-Efficient-COVID19-Diagnosis/blob/main/models/dino/pretrain_dino.sbatch) for reference.
