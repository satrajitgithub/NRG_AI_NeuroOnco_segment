#!/bin/bash

source $SCRIPT_ROOT/utils/bash_utils.sh
banner "Starting Segmentation"
echo "Checking for segmentationready flag..."
if [[ $1 == *xnat* ]];then 
  project=$2
  subject=$3
  session=$4

  segmentationready=$(get_custom_flag_xnat 'segmentationready' $project $subject $session)
  echo "segmentationready="$segmentationready
  
  if [[ $segmentationready != *"true"* ]];then 
    echo "Sorry! This session is NOT segmentationready"
  else
    python $SCRIPT_ROOT/wrapper_scripts/segmentation.py "${@:5}"
fi

else
  segmentationready=$(get_custom_flag)

  if [[ $segmentationready != *"true"* ]];then 
    echo "Sorry! This session is NOT segmentationready"
  else
    python $SCRIPT_ROOT/wrapper_scripts/segmentation.py "${@:2}"
  fi
fi