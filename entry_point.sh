#!/bin/bash

# ~~~~~~~~~~~~~~~~~~~~ Set paths ~~~~~~~~~~~~~~~~~~~~~~~~
echo "Updated: 10/06/2022"
# export TERM=xterm
export SCRIPT_ROOT=/NRG_AI_NeuroOnco_segment

print_help() {
  echo $"Error in input!
Usage (docker): docker run [-v <HOST_DIR>:<CONTAINER_DIR>] satrajit2012/nrg_ai_neuroonco_segment:v0 segmentation --docker [--evaluate] [--radiomics]
Usage (XNAT):  segmentation --xnat #PROJECT# #SUBJECT# #SESSION# [--evaluate] [--radiomics]"
}

case "$1" in
    segmentation) arg1_check=true;;
    *) print_help
            exit 1
esac

if [[ $arg1_check == *true* ]];then
  case "$2" in
      --docker) $SCRIPT_ROOT/routines/run_$1.sh "${@:2}";;
      --xnat) $SCRIPT_ROOT/routines/run_$1.sh "${@:2}";;
    *) print_help
              exit 1
  esac
fi