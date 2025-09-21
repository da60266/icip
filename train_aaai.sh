#!/usr/bin/env bash
declare -i lastfold=14
expname="8_27_eachimg3dmm"
declare -i epoch=10
datasetroot="/home/zsy/"
datasettarget="/ssddata/"
datasetname="MPIIFaceGaze900"
valdatasetname="MPIIFaceGaze900"

if [ ! -d ${datasettarget}${datasetname} ]
then
echo "Copying dataset from original folder to target folder..."
cp -r ${datasetroot}${datasetname} ${datasettarget}
cp -r ${datasetroot}${valdatasetname} ${datasettarget}
echo "OK."
echo ""
fi

declare -i foldini=0
if [ ${datasetname} != ${valdatasetname} ]
then
lastfold=-1
foldini=-1
fi

if [ ${lastfold} -lt 0 ]
then
foldini=-1
fi

for fold in $( seq $foldini $lastfold )
do

while [ ! -f "./okmark.txt" ];
do

declare -i pref=-1
declare -i count=0
for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
do
if [ ${i} -lt 20000 ]
then
pref=$count
fi
count=$(($count+1))
done

if [ $pref -lt 0 ]
then
echo "No CUDA Devices Available. Training stopped."
exit
fi

CUDA_VISIBLE_DEVICES=1 python trainer_aaai.py --data-root ${datasettarget}${datasetname} --val-data-root ${datasettarget}${valdatasetname} \
--batch-size-train 64 --batch-size-val 16 --num-workers 8 --exp-name ${expname} --fold-no $fold --visualize True --multitask=False --onlyeyeinput=False --augnum=1 --facesize=224 --usesmallface=False --targettype='gazedirection' \
 - train_base --epochs ${epoch} --lr 1e-4 --use-refined-depth False --fine-tune-headpose True\
 - log_experiment --separatebyexpname True --lastfold $lastfold --okmark okmark.txt\
 - end

done

rm ./okmark.txt

done

echo ""
echo "All fold completed."