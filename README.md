# MayaChallengeECML_PKDD-1st-place-solution
Repository contains code of 1st place solution (team Aksell) in the Discover the mysteries of the Maya @ ECML PKDD 2021- Integrated Image Segmentation Challenge hosted on codalab.org by Bias Variance Labs.
https://biasvariancelabs.github.io/maya_challenge/


./run.sh  #train models to detect aguada (encoder resnext101_32x4d_swsl)
./run2.sh   # train models to detect buildings and platforms (encoder resnext50_32x4d_swsl)
./run3.sh # train models to detect buildings and platforms (encoder resnext101_32x4d_swsl)

inference: UnetEnsemble[infer].ipynb
