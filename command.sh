python train.py --group=phototourism --model=barf --yaml=barf_phototourism --name=brandenburg_garf --data.scene=brandenburg_gate --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True
python train.py --group=phototourism --model=barf --yaml=barf_phototourism --name=sacre_garf --data.scene=sacre_coeur --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True
python train.py --group=phototourism --model=barf --yaml=barf_phototourism --name=trevi_garf --data.scene=trevi_fountain --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True
python train.py --group=phototourism --model=barf --yaml=barf_phototourism --name=taj_garf --data.scene=taj_mahal --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True
python evaluate.py --group=phototourism --model=barf --yaml=barf_phototourism --name=brandenburg_garf --data.scene=brandenburg_gate --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True --resume
python evaluate.py --group=phototourism --model=barf --yaml=barf_phototourism --name=sacre_garf --data.scene=sacre_coeur --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True --resume
python evaluate.py --group=phototourism --model=barf --yaml=barf_phototourism --name=trevi_garf --data.scene=trevi_fountain --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True --resume
python evaluate.py --group=phototourism --model=barf --yaml=barf_phototourism --name=taj_garf --data.scene=taj_mahal --visdom=False --feature.encode=False --transient.encode=False --arch.garf=True --resume