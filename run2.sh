#train models to detect buildings and platforms (encoder resnext50_32x4d_swsl)
for i in {0..3};do
  python train2.py --fold $i --nfolds 4 --batch_size 16 --epochs 50 --lr 2e-3  --workers 4 --encoder resnext50_32x4d_swsl --prefix building  --out_dir buildingV1
done  
for i in {0..3};do
  python train2.py --fold $i --nfolds 4 --batch_size 16 --epochs 50 --lr 2e-3  --workers 4 --encoder resnext50_32x4d_swsl --prefix platform  --out_dir platformV1	
done
