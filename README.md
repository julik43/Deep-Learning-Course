# Deep-Learning-Course
This git is part of the course Deep Learning at the UNAM, teached by the PhD. Gibran Fuentes-Pineda.

This git has the aim to recreate the experiment of "Unsupervised Feature Learning Based on Deep Models for Environmental Audio Tagging" and to create a new convolutional network to work with the DCASE 2016 task 4 database.

# To recreate the experiment

To recreate the experiment of: Unsupervised Feature Learning Based on Deep Models for Environmental Audio Tagging
Available in: https://github.com/yongxuUSTC/aDAE_DNN_audio_tagging

You need to follow the next steps:

1. Download the "dcase2016_task4-master" database	

	This database is available in: http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging

	or going to: http://www.cs.tut.fi/sgn/arg/dcase2016/index and clicking on Domestic Audio Tagging.

	Once you downloaded the database you go to the baseline folder and follow the steps of the readme.

	Be aware of the localization of the dataset and where is going to be located the audio information, normally it will be: dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home/chunks

2. Go to the folder aDAE_DNN to continue the recreation of the experiment.

	Download the Hat library from (it must be this version of the library, with a different one it would not work): https://codeload.github.com/qiuqiangkong/Hat/zip/9f1d088f6be7bf2e159e521c5dac46c800c86ff2

3. Download the original code from the git hub of the creators:

	Download the code from: https://github.com/yongxuUSTC/aDAE_DNN_audio_tagging

4. Modify or replace the codes of the creator with the ones you can find in here in the folder aDAE_DNN
	
	Note: the genFea_DAE_keras.py code is not found on the original github but in the deleted codes.

5. On the next codes change "your_dir" with your correct direction for the desired folder, it is explain above every line which is the folder you must be looking for:

	* config_fb40.py --> This file configurates everything to be used in the DNN architecture and the generation of the features.
	* prepare_data_1ch_MFC.py --> This file is the one to load the data already processed (the characteristics) and to actually extract the characteristics.
	* main_DAE_keras.py --> This is the model to generate the characteristics with the symetric or asymetric autoencoder, here the characteristics are still not generated, but the model is trained.
	* genFea_DAE_keras_v2.py --> this is the file to create the files with the characteristics with the autoencoders trained in the item before.
	* main_DNN_keras_EVAL_fb40.py --> This is the file were the DNN is generated and trained.
	* recognize_dnn_1ch_keras_NAT_eva.py --> This is the file to get out all the metrics about the models.

6. To generate the MFCC
	* Once you have done all the previous steps, go to the config_fb40.py and uncomment the line:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_MFCC'	 	# For MFCC 24 
	
	and comment it for the rest of the options.
	
	Then go to the prepare_data_1ch_MFC.py and uncomment the section of the main that says:
	
	To create the MFCC Characteristics
	
	and comment it for the other options, and then run that code from a terminal by doing python prepare_data_1ch_MFC.py

7. To generate the MFB
	* Once you have done all the previous steps, go to the config_fb40.py and uncomment the line:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
	
	and comment it for the rest of the options.
	
	Then go to the prepare_data_1ch_MFC.py and uncomment the section of the main that says:
	
	To generate the MFB characteristic
	
	and comment it for the other options, and then run that code from a terminal by doing python prepare_data_1ch_MFC.py

Note: From here on, all the DAE and DNN models are stored in the folder Md, to avoid losing your models, we encourage you to change the name of the folder everytime you run a model and create a new folder Md for the next experiments.

Note: to the generation of the symetric and asymetric characteristics we only used the MFB characteristics, this can also be done using the MFCC characteritics in the same way

8. To generate the symetric characteristics
	* Once you have done all the previous steps, go to the config_fb40.py and uncomment the line:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
	
	and comment it for the rest of the options.
	
	Then go to the main_DAE_keras.py and uncomment the section that says:
	
	Model for the symetric
	
	For symetric
	
	and comment it for the other options, and then run that code from a terminal by doing python main_DAE_keras.py

	After this create a folder with the name htk_syDAE on the next location:
	
	DCASE2016_task4_scrap_1ch_mfcc/Fe
	
	go to the config_fb40.py and uncomment the line:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_syDAE'		# for symetric
	
	and comment it for the rest of the options.
	
	Then go to the genFea_DAE_keras_v2.py and uncomment the section that says:
	
	For symetric
	
	and comment it for the other options. Then you must change your root to the model that was generated by the main_DAE_keras on the previous step, and then run the code from a terminal by doing python genFea_DAE_keras_v2.py

9. To generate the asymetric characteristics
	* Once you have done all the previous steps, go to the config_fb40.py and uncomment the line:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
	
	and comment it for the rest of the options.
	
	Then go to the main_DAE_keras.py and uncomment the section that says:
	
	Model for the asymetric
	
	For asymetric
	
	and comment it for the other options, and then run that code from a terminal by doing python main_DAE_keras.py

	After this create a folder with the name htk_asyDAE on the next location:
	
	DCASE2016_task4_scrap_1ch_mfcc/Fe
	
	go to the config_fb40.py and uncomment the line:
	
	#dev_fe_mel_fd = scrap_fd + '/Fe/htk_asyDAE'	# For asymetric
	
	and comment it for the rest of the options.
	
	Then go to the genFea_DAE_keras_v2.py and uncomment the sections that says:
	
	For asymetric
	
	and comment it for the other options. Then you must change your root to the model that was generated by the main_DAE_keras on the previous step, and then run the code from a terminal by doing python genFea_DAE_keras_v2.py

10. To run the DNN with any of the previous characteristics generated

	* Go to the config_fb40.py and uncomment the line of the desired characteristics, if for example is MFB40:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
	
	and comment it for the rest of the options.
	
	Then go to the main_DNN_keras_EVAL_fb40.py and uncomment the section of the desired characteristics, if for example is MFB40
	
	For MFB 40 
	
	and comment it for the rest of the options.
	
	Like this recreation of the experiments is done using cross validation, you must choose a fold, that is one of the partitions of the dataset, for this we have from 0 to 4 folds.
	
	Then run the code from a terminal by doing python main_DNN_keras_EVAL_fb40.py

	Please remember the note about the folder Md.

11. To evaluate the trained models

	* Go to the config_fb40.py and uncomment the line of the desired characteristics, if for example is MFB40:
	
	dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
	
	and comment it for the rest of the options.
	
	Then go to the recognize_dnn_1ch_keras_NAT_eva.py and uncomment the section of the desired characteristics, if for example is MFB40 
	
	For MFB 40 
	
	and comment it for the rest of the options.
	
	Then you must load the model that was generated by main_DNN_keras_EVAL_fb40.py, in this code is an example for every type of characteristic and for every fold.
	
	Then run the code from a terminal by doing python recognize_dnn_1ch_keras_NAT_eva.py


# To create a new CNN

To create a new convolutional network to work with the DCASE 2016 task 4 database.

1. Download the "dcase2016_task4-master" database	

	This database is available in: http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging

	or going to: http://www.cs.tut.fi/sgn/arg/dcase2016/index and clicking on Domestic Audio Tagging

	Once you downloaded the database you go to the baseline folder and follow the steps of the readme.

	Be aware of the localization of the dataset and where is going to be located the audio information, normally it will be: dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home/chunks

2. To run the model Xavier_GRAD_SPECTROGRAM_4S.py or Xavier_RMS_SPECTROGRAM_4S.py

	You need four files:
	
	* train_speakers.txt --> a list of audio files for training, basically is the same founded in dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home but with the proper route
	
	* valid_speakers.txt --> a list of audio files for validation, basically is the same founded in dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home but with the proper route
	
	* train_speakers_tag.txt --> a list of csv files containing by audio the information of classification for the training audios, if the database was downloaded as in step one, they are located in dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home/chunks, the same place of the audios, and have the same name of the audio but with a csv extension.
	
	* valid_speakers_tag.txt --> a list of csv files containing by audio the information of classification for the validation audios, if the database was downloaded as in step one, they are located in dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home/chunks, the same place of the audios, and have the same name of the audio but with a csv extension.

	Once you have the file you can run the code from a terminal by doing python Xavier_GRAD_SPECTROGRAM_4S.py or python Xavier_RMS_SPECTROGRAM_4S.py

3. Evaluation

	To evaluate the model, you must go to evaluation.py and write the proper routes to the files exported in the step 2, in the lines:
	
	For grad
	
    list_audios = np.load('GRAD_Results/list_audios.npy')
    
    y_pred = np.load('GRAD_Results/y_pred.npy')
    
    y_real = np.load('GRAD_Results/y_real.npy')
    
    y_sig = np.load('GRAD_Results/y_sig.npy')

    Once you have done this, you can run the code from a terminal by doing python evaluation.py and then you will have your statistics.

