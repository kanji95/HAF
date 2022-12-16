<<<<<<< HEAD
python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 250 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/species/seed1 --seed 10 --taxonomy species
=======
python main.py --start training --arch resnet18 --batch-size 256 --epochs 250 --loss cross-entropy --optimizer adamw --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed0 --seed 0 --taxonomy genus
>>>>>>> 3ac9263e03c099cb0be12ca5bd2b71f452711c3c
