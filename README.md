# [iROP](http://i-rop.github.io/) Image Quality Assessment Tool

This is a tool designed for the automated assessment of retinal fundus image quality of images collected from premature babies who underwent routine retinopathy of prematurity screening examinations.

## Files
* **kfold_cnn.py**
    - `trainCNN` performs k-fold cross-validation on a set of ROP images (graded as either Acceptable Quality or Not Acceptable Quality). For each fold, the "best" model (defined as that achieving the lowest loss thus far in the training process) is saved into a folder. Users must manually extract the best model from that folder.
    - `KFoldROC` produces ROC curves for each of the k models trained by `trainCNN`.


## Abstract

#### Purpose
Accurate image-based ophthalmic diagnosis relies on fundus image clarity. This has important implications for the quality of ophthalmic diagnoses and for emerging methods such as telemedicine and computer-based image analysis. The purpose of this study was to implement a deep convolutional neural network (CNN) for automated assessment of fundus image quality in retinopathy of prematurity (ROP).

#### Participants
Retinal fundus images were collected from preterm infants during routine ROP screenings.

#### Methods
Six thousand one hundred thirty-nine retinal fundus images were collected from 9 academic institutions. Each image was graded for quality (acceptable quality [AQ], possibly acceptable quality [PAQ], or not acceptable quality [NAQ]) by 3 independent experts. Quality was defined as the ability to assess an image confidently for the presence of ROP. Of the 6139 images, NAQ, PAQ, and AQ images represented 5.6%, 43.6%, and 50.8% of the image set, respectively. Because of low representation of NAQ images in the data set, images labeled NAQ were grouped into the PAQ category, and a binary CNN classifier was trained using 5-fold cross-validation on 4000 images. A test set of 2109 images was held out for final model evaluation. Additionally, 30 images were ranked from worst to best quality by 6 experts via pairwise comparisons, and the CNN’s ability to rank quality, regardless of quality classification, was assessed.

#### Main Outcome Measures
The CNN performance was evaluated using area under the receiver operating characteristic curve (AUC). A Spearman’s rank correlation was calculated to evaluate the overall ability of the CNN to rank images from worst to best quality as compared with experts.

#### Results
The mean AUC for 5-fold cross-validation was 0.958 (standard deviation, 0.005) for the diagnosis of AQ versus PAQ images. The AUC was 0.965 for the test set. The Spearman’s rank correlation coefficient on the set of 30 images was 0.90 as compared with the overall expert consensus ranking.

#### Conclusions
This model accurately assessed retinal fundus image quality in a comparable manner with that of experts. This fully automated model has potential for application in clinical settings, telemedicine, and computer-based image analysis in ROP and for generalizability to other ophthalmic diseases.


## Related Publications

* **Coyner AS**, Swan R, Campbell JP, Ostmo S, Brown JM, Kalpathy-Cramer J, Kim SJ, Jonas KE, Chan RVP, Chiang MF. Automated Fundus Image Quality Assessment in Retinopathy of Prematurity Using Deep Convolutional Neural Networks. Ophthalmol Retina. 2019 May;3(5):444-450. doi: 10.1016/j.oret.2019.01.015.
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.oret.2019.01.015-blue.svg)](https://doi.org/10.1016/j.oret.2019.01.015)

* **Coyner AS**, Swan R, Brown JM, Kalpathy-Cramer J, Kim SJ, Campbell JP, Jonas KE, Ostmo S, Chan RVP, Chiang MF. Deep Learning for Image Quality Assessment of Fundus Images in Retinopathy of Prematurity. AMIA Annu Symp Proc. 2018 Dec 5;2018:1224-1232.
[![PMCID](https://img.shields.io/badge/PMCID-PMC6371336-green.svg)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371336/)
