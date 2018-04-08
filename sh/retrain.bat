@echo off

REM 224, 192, 160 or 128
SET IMAGE_SIZE=224

REM 1.0, 0.75, 0.50 or 0.25
SET WIDTH=1.0

REM adam or sgd
SET "OPTIMIZER=sgd"
SET LEARNING_RATE=0.01
SET BATCH_SIZE=128
SET TEST_PERC=5
SET STEPS=500
SET LABEL="bs_%BATCH_SIZE%-lr_%LEARNING_RATE%-opt_%OPTIMIZER%"

SET ARCHITECTURE="mobilenet_v1_%WIDTH%_%IMAGE_SIZE%"

SET TF_COMMAND=python -m scripts.retrain ^
  --bottleneck_dir=tf_files/bottlenecks ^
  --model_dir=tf_files/models ^
  --summaries_dir=tf_files/training_summaries/%LABEL% ^
  --output_graph="tf_files/retrained_graph_%WIDTH%.pb" ^
  --output_labels=tf_files/retrained_labels.txt ^
  --architecture=%ARCHITECTURE% ^
  --image_dir=tf_files/flower_photos ^
  --testing_percentage=%TEST_PERC% ^
  --how_many_training_steps=%STEPS% ^
  --learning_rate=%LEARNING_RATE% ^
  --train_batch_size=%BATCH_SIZE% ^
  --optimizer_name=%OPTIMIZER%

CALL %TF_COMMAND%

@echo on