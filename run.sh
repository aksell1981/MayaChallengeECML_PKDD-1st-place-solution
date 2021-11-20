#train models to detect aguada
for i in {0..4};do
  python train.py --fold $i --nfolds 5 --batch_size 16 --epochs 50 --lr 1e-4  --workers 16 --prefix aguada  --encoder resnext101_32x4d_swsl 
done  

