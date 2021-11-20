#train models to detect buildings and platforms (encoder resnext101_32x4d_swsl)
for i in {0..3};do
  python train2.py --fold $i --nfolds 4 --batch_size 16 --epochs 50 --lr 1e-3  --workers 4 --encoder resnext101_32x4d_swsl --prefix building  --out_dir buildingV2
done  
for i in {0..3};do
  python train2.py --fold $i --nfolds 4 --batch_size 16 --epochs 50 --lr 1e-3  --workers 4 --encoder resnext101_32x4d_swsl --prefix platform  --out_dir platformV2	
done
