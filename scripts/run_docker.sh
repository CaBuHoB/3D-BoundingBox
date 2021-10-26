#!/bin/bash

RUN_COMMAND="python"

case ${MODE,,} in

  eval)
    RUN_COMMAND="${RUN_COMMAND} Run.py"
    ;;

  eval_no_yolo)
    RUN_COMMAND="${RUN_COMMAND} Run_no_yolo.py"
    ;;

  train)
    RUN_COMMAND="${RUN_COMMAND} Train.py"
    ;;
  *)
    echo "There are no instructions for the mod ${MODE^^}."
    echo "Please choose from the possible: eval, eval_no_yolo, train."
    exit 1
    ;;
esac

if [[ -n "$DEVICE" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --device=${DEVICE}"
fi

if [[ -n "$WEIGHTS_PATH" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --weights-path=${WEIGHTS_PATH}"
fi

if [[ -n "$DATASET_PATH" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --dataset-path=${DATASET_PATH}"
fi

if [[ -n "$CALIB_PATH" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --calib-path=${CALIB_PATH}"
fi

if [[ "${MODE,,}" == "eval" || "${MODE,,}" == "eval_no_yolo" ]] && [[ -n "$IMWRITE" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --imwrite=${IMWRITE}"
fi

if [[ "${MODE,,}" == "eval" || "${MODE,,}" == "eval_no_yolo" ]] && [[ -n "$OUTPUT_DIR" ]]; then
    RUN_COMMAND="${RUN_COMMAND} --output-dir=${OUTPUT_DIR}"
fi

if [[ "${MODE,,}" == "eval" ]] && [[ -n "$VIDEO" && "$VIDEO" -eq 1 ]]; then
    RUN_COMMAND="${RUN_COMMAND} --video"
fi

if [[ "${MODE,,}" == "eval" ]] && [[ -n "$SHOW_YOLO" && "$SHOW_YOLO" -eq 1 ]]; then
    RUN_COMMAND="${RUN_COMMAND} --show-yolo"
fi

if [[ "${MODE,,}" == "eval" ]] && [[ -n "$HIDE_DEBUG" && "$HIDE_DEBUG" -eq 1 ]]; then
    RUN_COMMAND="${RUN_COMMAND} --hide-debug"
fi

echo $RUN_COMMAND 
$RUN_COMMAND