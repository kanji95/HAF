python main.py --start training --arch resnet18 --batch-size 256 --epochs 250 --loss cross-entropy --optimizer adamw --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed0 --seed 0 --taxonomy genus