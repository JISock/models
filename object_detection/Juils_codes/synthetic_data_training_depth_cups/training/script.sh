
# training(execute from the tensorflow root folder)
python train.py --logtostderr --pipeline_config_path=./Juils_codes/synthetic_data_training_depth_cups/training/faster_rcnn_inception_resnet_v2_atrous_synthetic.config --train_dir=./Juils_codes/synthetic_data_training_depth_cups/training/train

# freeze model(export_inference_graph.py is in the upper dir)

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./Juils_codes/synthetic_data_training_depth_cups/Training/faster_rcnn_inception_resnet_v2_atrous_synthetic.config --trained_checkpoint_prefix ./Juils_codes/synthetic_data_training_depth_cups/training/train/model.ckpt-2174 --output_directory ./Juils_codes/training/synthetic_data_training_depth_cups/train/synthetic_inference_graph_2174.pb

# testing
