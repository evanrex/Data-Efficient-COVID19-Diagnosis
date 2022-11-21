import os
import shutil
import pandas as pd

COVIDXCTPATH = '~/research_project/covidx-ct/' #'/tmp/covidx-ct/'
DESTINATIONDIR='~/research_project/Covidx-CT/' #"/tmp/Covidx-CT/"

def input_data():
    covidxct_path = COVIDXCTPATH#'/tmp/covidx-ct/'
    image_path = covidxct_path+'3A_images/'
    train_df = pd.read_csv(covidxct_path+'train_COVIDx_CT-3A.txt', sep=" ", header=None)
    train_df.columns=['filename', 'label', 'xmin','ymin','xmax','ymax']
    train_df=train_df.drop(['xmin', 'ymin','xmax', 'ymax'], axis=1 )

    val_df = pd.read_csv(covidxct_path+'val_COVIDx_CT-3A.txt', sep=" ", header=None)
    val_df.columns=['filename', 'label', 'xmin','ymin','xmax','ymax']
    val_df=val_df.drop(['xmin', 'ymin','xmax', 'ymax'], axis=1 )

    test_df = pd.read_csv(covidxct_path+'test_COVIDx_CT-3A.txt', sep=" ", header=None)
    test_df.columns=['filename', 'label', 'xmin','ymin','xmax','ymax']
    test_df=test_df.drop(['xmin', 'ymin','xmax', 'ymax'], axis=1 )

    # train_df['filename'] = image_path+train_df['filename']
    # val_df['filename'] = image_path+val_df['filename']
    # test_df['filename'] = image_path + test_df['filename']

    # Here, we label pneumonia as normal since we are only trying to label for covid negative/positive
    train_df['label'] = train_df['label'].replace([1,2],[0,1])
    val_df['label'] = val_df['label'].replace([1,2],[0,1])
    test_df['label'] = test_df['label'].replace([1,2],[0,1])

    labels={0:'Normal',1:'COVID-19'}
    class_names=['Normal','COVID-19']

    train_df['label_n']=[labels[b] for b in train_df['label']]
    val_df['label_n']=[labels[b] for b in val_df['label']]
    test_df['label_n']=[labels[b] for b in test_df['label']]

    return train_df,val_df,test_df

def main():
    print('start')
    # Data location
    covidxct_path= '/home-mscluster/erex/research_project/covidx-ct/' #COVIDXCTPATH #'/tmp/covidx-ct/'
    IMAGE_DIR = covidxct_path+'3A_images/'
    dst_dir = '/home-mscluster/erex/research_project/Covidx-CT/' #DESTINATIONDIR #"/tmp/Covidx-CT/"

    train_df,val_df,test_df=input_data()
    print('csv read in')

    print(len(train_df))
    print(len(val_df))
    print(len(test_df))

    print(train_df.head())

    train_positive = train_df[train_df['label']==1]
    train_negative = train_df[train_df['label']==0]

    for filename in train_positive.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'train/positive/'+filename, copy_function = shutil.copytree)
    
    for filename in train_negative.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'train/negative/'+filename, copy_function = shutil.copytree)

    print('train moved')

    val_positive = val_df[val_df['label']==1]
    val_negative = val_df[val_df['label']==0]

    for filename in val_positive.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'validation/positive/'+filename, copy_function = shutil.copytree)
    
    for filename in val_negative.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'validation/negative/'+filename, copy_function = shutil.copytree)

    print('validation moved')

    test_positive = test_df[test_df['label']==1]
    test_negative = test_df[test_df['label']==0]

    for filename in test_positive.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'test/positive/'+filename, copy_function = shutil.copytree)
    
    for filename in test_negative.filename:
        shutil.move(IMAGE_DIR+filename, dst_dir+'test/negative/'+filename, copy_function = shutil.copytree)

    print('done')

if __name__ == "__main__":
    main()