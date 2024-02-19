The data files 'fine_tuning_data_detic.csv' and 'fine_tuning_data_yolov5.csv' are the result of the preparation and
filtration after performing below steps:

- Generate the captions for all the images.
- Delete all samples with corrupted or rubbish data. (Please refer to the report for details)
- Run object detection models ('yolov5' and 'detic') and generate the corresponding objects for the images corresponding to the remaining samples.
- Convert all the question, answer, caption, objects together with the system prompt into the desired template for all
  the samples (Please refer to the report for the detailed template design).