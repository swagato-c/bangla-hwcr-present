import argparse
from fastai import *
from fastai.vision import *
import os
import time
import shutil
import torch


def post_train_processing(learner,isUnfreezed,no_of_epochs,lrl,lrh,path,export_inference,custom_suffix=''):
	tmstp=time.strftime("%d%m%Y%H%M%S",time.localtime())
	save_name="{}_{}_e{}_lrl{}_lrh{}_{}".format(tmstp,
	  'unfreezed' if isUnfreezed else 'freezed',no_of_epochs,lrl,lrh,custom_suffix)
	learner.save(save_name)
	print('Model saved to {}/{}.pth.'.format(path,save_name))
	learner.recorder.plot_losses()
	learner.recorder.plot_metrics()
	learner.recorder.plot_lr()
	shutil.copyfile("{}/history.csv".format(path),"{}/{}_history.csv".format(path,save_name))
	#shutil.copyfile("{}/loss-history.csv".format(path),"{}/{}_loss-history.csv".format(path,save_name))
	print('Saved files {0}/{1}_history.csv'.format(path,save_name))
	#!cp {path}/history.csv {path}/{save_name}_history.csv
	#!cp {path}/loss-history.csv {path}/{save_name}_loss-history.csv
	interp = ClassificationInterpretation.from_learner(learner)
	#interp.plot_top_losses(9)
	mat = interp.confusion_matrix()
	c = interp.data.classes
	pd.DataFrame(mat,index=c,columns=c).to_csv("{}/{}_confusion_matrix.csv".format(path,save_name),index=True)
	print('Saved file {}/{}_confusion_matrix.csv'.format(path,save_name))
	if export_inference:
		learner.export()
		print('Saved file {}/export.pkl for inference'.format(path))


def train(input_file_name,images_path,pretrained,model_file,freeze_layers,lrl,lrh,sz,bs,epochs,custom_save_suffix,export_inference,arch=models.resnet50):
		src = ImageItemList.from_csv(images_path,input_file_name,cols='x').split_from_df(col='is_valid').label_from_df(cols='y')
		data = src.transform(get_transforms(do_flip=False), size=sz).databunch(bs=bs).normalize(imagenet_stats)
		if model_file:
			pretrained=False
		learn = create_cnn(data,arch,metrics=[error_rate,accuracy],callback_fns=[callbacks.CSVLogger],pretrained=pretrained)
		if model_file:
			head,tail=os.path.split(model_file)
			learn.load(model_file.split('.')[0])
			print('Loaded model file')
			learn.data=data
			print('Loaded data into model')
		#print(len(learn.layer_groups))
		#print(data.valid_ds)
		#print(data.train_ds)
		learn.freeze() if freeze_layers else learn.unfreeze()
		#print(epochs)
		#print(lrl)
		#print(lrh)
		print('Starting ')
		if freeze_layers:
			learn.fit_one_cycle(epochs,slice(lrl))
		else:
			learn.fit_one_cycle(epochs,slice(lrl,lrh))
		post_train_processing(learn,not freeze_layers,epochs,lrl,'' if freeze_layers else lrh,images_path,export_inference,'' if not custom_save_suffix else custom_save_suffix)
		del learn.model
		del learn
		torch.cuda.empty_cache()
		print('Post train processing complete')
	





def main():
	parser=argparse.ArgumentParser()
	parser.add_argument('-i','--input_file_name',help='this should be the name of the csv file containing all the file names (col_name: x), the ground truth (col_name: y) and whether each file belongs to test or validation (col_name:is_valid)',required=True)
	parser.add_argument('-im','--images_path',help='path of the images folder...the path present in the csv file will be appended to this path for each image...even the output files are stored in this directory',required=True)
	parser.add_argument('-p','--pretrained',help='If --model_file or -m is not used and --pretrained or -p flag is used, then a pretrained weights trained on ImageNet is downloaded and used...if --pretrained or -p is not specified and --model_file or -m is also not mentioned, random initialization is done...if --model_file or -m is mentioned, this value is ignored',required=False,action='store_true')
	parser.add_argument('-m','--model_file_name',help='Name of Model File...should be present at --images_path/models/ directory ',required=False)
	parser.add_argument('-f','--freeze_layers',help='If used, freezes the non-FC layers',action='store_true')
	parser.add_argument('-lrl','--lower_lr',help='Lower learning rate for training...if -f or --freeze_layers flag is used, then -lrh or --higher_lr is not required',required=True,type=float)
	parser.add_argument('-lrh','--higher_lr',help='Higher lr for training...only required if -f or --freeze_layers flag is not used',required=False,type=float)
	parser.add_argument('-sz','--image_size',help='Image size to be used for training',required=True,type=int)
	parser.add_argument('-bs','--batch_size',help='Batch size to be used',required=True,type=int)
	parser.add_argument('-e','--epochs',help='No of epochs to train for',required=True,type=int)
	parser.add_argument('-s','--custom_save_suffix',help='Custom suffix to be added at the end of the name of files to be stored as output',required=False)
	parser.add_argument('-ei','--export_inference',help='Export model for inference (.pkl file)...the file is saved as export.pkl under the directory given in -im or --images_path/models/ directory',action='store_true')
	args=parser.parse_args()
	train(args.input_file_name,args.images_path,args.pretrained,args.model_file_name,args.freeze_layers,args.lower_lr,args.higher_lr,args.image_size,args.batch_size,args.epochs,args.custom_save_suffix,args.export_inference)
if __name__=='__main__':
	main()

