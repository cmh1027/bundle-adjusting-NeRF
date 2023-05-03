. ./command/var.sh
DEVICE=$1
SCENE=$2
CUDA_VISIBLE_DEVICES=${DEVICE} \
python train.py --group=blender --model=barf --yaml=barf_blender --visdom=false \
--name=${SCENE} --data.scene=${SCENE} \
--barf_c2f=[${COARSE},${FINE}] --camera.noise=${NOISE} \
--transient.encode=False --feature.encode=False \
--max_iter=${MAX_ITER}

CUDA_VISIBLE_DEVICES=${DEVICE} \
python train.py --group=blender --model=barf --yaml=barf_blender --visdom=false \
--name=${SCENE}_transient --data.scene=${SCENE} \
--barf_c2f=[${COARSE},${FINE}] --camera.noise=${NOISE} \
--transient.encode=True --feature.encode=False \
--max_iter=${MAX_ITER}

CUDA_VISIBLE_DEVICES=${DEVICE} \
python train.py --group=blender --model=barf --yaml=barf_blender --visdom=false \
--name=${SCENE}_feature --data.scene=${SCENE} \
--barf_c2f=[${COARSE},${FINE}] --camera.noise=${NOISE} \
--transient.encode=False --feature.encode=True \
--max_iter=${MAX_ITER}

CUDA_VISIBLE_DEVICES=${DEVICE} \
python train.py --group=blender --model=barf --yaml=barf_blender --visdom=false \
--name=${SCENE}_transient_feature --data.scene=${SCENE} \
--barf_c2f=[${COARSE},${FINE}] --camera.noise=${NOISE} \
--transient.encode=True --feature.encode=True \
--max_iter=${MAX_ITER}