
# development
ori_dev_root = 'your_dir/dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home'

# Folder where you can find the audio chunks
dev_wav_fd = 'your_dir/dcase2016_task4-master/baseline/data/CHiMEHome-audiotag-development/chime_home/chunks'

# Folder where are going to be located the extracted features 
scrap_fd = "your_dir/DCASE2016_task4_scrap_1ch_mfcc"
#dev_fe_mel_fd = scrap_fd + '/Fe/htk_MFCC'	 	# For MFCC 24
#dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40' 		# For MFB40
#dev_fe_mel_fd = scrap_fd + '/Fe/htk_asyDAE'	# For asymetric
dev_fe_mel_fd = scrap_fd + '/Fe/htk_syDAE'		# for symetric

# Where is going to be located the path for the chunks roots
dev_cv_csv_path = ori_dev_root + '/development_chunks_refined_crossval_dcase2016.csv'

# I never used this
# evaluation
'''
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'
'''

# Labels for the different classes, this is used in the recognition process
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000
win = 320
