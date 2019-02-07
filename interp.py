import argparse
from fastai import *
from fastai.vision import *
import os

def pred_single(input_file,weight_file):
	head,tail=os.path.split(weight_file)
	input_img=open_image(input_file)
	learn=load_learner(head,tail)
	return learn.predict(input_img)

def preds(input_file,weight_file,image_path,output_file,col_name):
	csv_df=pd.read_csv(input_file)
	head,tail=os.path.split(weight_file)
	learn=load_learner(head,tail,test=ImageItemList.from_df(df=csv_df,path=image_path,cols=col_name))
	#print(learn.data.test_ds.x.items)
	preds,y = learn.get_preds(ds_type=DatasetType.Test)
	if type(preds)==torch.Tensor:
		preds=preds.numpy()
	if type(y)==torch.Tensor:
		y=y.numpy()
	#print(y)
	#print(preds.shape)
	#print(y.shape)
	result_df=pd.DataFrame(data=preds,columns=learn.data.train_ds.y.classes)
	result_df['input_file']=pd.Series(learn.data.test_ds.x.items)
	#print(np.argmax(preds,axis=1).shape)
	result_df['pred']=pd.Series(np.argmax(preds,axis=1)).apply(lambda x:learn.data.train_ds.y.classes[x])
	print("Saving file to {}".format(output_file))
	result_df.to_csv(output_file,index=False,columns=['input_file']+learn.data.train_ds.y.classes+['pred'])
	return result_df




def main():
	parser=argparse.ArgumentParser()
	parser.add_argument('-f','--input_file',help='path of file to be interpreted...if used wih -s or --pred_single, then this should be a csv file containing all the file names that has to be predicted',required=True)
	parser.add_argument('-w','--weight_file',help='Path of Weight File',required=True)
	parser.add_argument('-m','--pred_multiple',help='Predict multiple images...if used,input file should be a csv file containing all the image file paths to be predicted and if not used, then the input_file should be an image file',action='store_true')
	parser.add_argument('-o','--csv_output',help='if -m or --pred_multiple is used, then this argument has to be specified containing the output location of the csv file which will contain the predictions')
	parser.add_argument('-i','--image_path',help='if -m or --pred_multiple is used, provide the base path of the images.')
	parser.add_argument('-c','--csv_column',help='if -m or --pred_multiple is used, then the column name of csv_file is mentioned here.')
	args=parser.parse_args()
	if not args.pred_multiple:
		print(pred_single(args.input_file,args.weight_file))
	else:
		pred_df=preds(args.input_file,args.weight_file,args.image_path,args.csv_output,args.csv_column)

if __name__=='__main__':
	main()

