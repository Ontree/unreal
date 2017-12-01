from train.experience import Experience, ExperienceFrame
import pickle
import json
from PIL import Image
from scipy.misc import toimage

f = open('/media/bighdd5/minghai1/capstone/results3/MsPacmanNoFrameskip-v0_fsr_10_ae/res.pkl', 'rb')
data = pickle.load(f)

# generate image of 'state'
for i in range(len(data)):
	for j in range(4):
		#img = Image.fromarray(data[i]['states'][j], 'RGB')
		img = toimage(data[i]['states'][j])
		img.save('img_state/' + str(i) + '_' + str(j)+ '.png')
	img = toimage(data[i]['next_frame_prediction'][0])
	img.save('img_next_frame_prediction/' + str(i) + '.png')
	img = toimage(data[i]['next_frame_ground_truth'].state)
	img.save('img_next_frame_ground_truth/' + str(i) + '.png')
	

def get_max_index(a):
	b = a.tolist()[0]
	tmp_max = -10000
	tmp_idx = 0
	for i in range(len(b)):
		if b[i]>tmp_max:
			tmp_max = b[i]
			tmp_idx = i
	return tmp_idx
rst = []
a = [0,1,-1]
for i in range(len(data)):
	tmp = {}
	tmp['pre_reward'] = get_max_index(data[i]['next_reward_prediction'])
	tmp['ground_truth_last_act'] = int(data[i]['next_frame_ground_truth'].last_action)
	tmp['action'] = data[i]['action'][0].tolist().index(1)
	tmp['true_reward'] = data[i]['target_reward'][0].index(1)
	tmp['step'] = data[i]['step']
	rst.append(tmp)
print (json.dumps(rst))
	
f.close()






