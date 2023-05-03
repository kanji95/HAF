# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed3 --seed 30 --taxonomy genus --pretrained

python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/order/seed0 --seed 0 --taxonomy order --pretrained

# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/semi_super/cl_10/seed0 --seed 0 --taxonomy species --pretrained --class_limit 10
