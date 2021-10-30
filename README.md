# MVS-Light (WIP)

We propos a method for learning multi-view stereo without 3D cost regularization, where we use similarity between warped views to inform depth estimation.
Our architecture is shown below:

The framework and data loading utils of of this repo are taken from the CasMVSNet (link here) repo and modified for training and testing MVSLight.
Differeiable warping is implemented as in PatchMatchNet and we modfiy the similarity calculation for per view weights. 


### Data set up 
Similarly to other methods, we train our model on the DTU dataset.
Set up instructions for this dataset are taken for the CasMVSNet repo (link here).
* First, download [DTU dataset](https://roboimagedata.compute.dtu.dk/). For convenience, can download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.

```                
├── Cameras    
├── Depths
├── Depths_raw   
├── Rectified
├── Cameras                               
             
```
### Training
* In ``train.sh``, set ``MVS_TRAINING`` as $MVS_TRANING
* Train MVSLight on a single GPU:
```
export save_results_dir="./checkpoints"
./train.sh $save_results_dir  --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3
```

## Testing and Fusion
* Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the $TESTPATH folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* In ``test.sh``, set ``TESTPATH`` as $TESTPATH.
* Set ``CKPT_FILE``  as your checkpoint file, you also can download my [pretrained model](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/48_32_8-4-2-1_dlossw-0.5-1.0-2.0/casmvsnet.ckpt).
* Test CasMVSNet and Fusion( default is provided by [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch)): 
```
export save_results_dir="./outputs"
./test.sh  $CKPT_FILE --outdir $save_results_dir  --interval_scale 1.06
```
## Results on DTU
|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| MVSLight        | 0.396  | 0.527  | 0.462    |


## Results on Tanks and Temples benchmark

| Mean   | Family | Francis | Horse  | Lighthouse | M60    | Panther | Playground | Train |
|--------|--------|---------|--------|------------|--------|---------|------------|-------|
| 56.42  | 76.36  | 58.45   | 46.20  | 55.53	  | 56.11  | 54.02   | 58.17	  | 46.56 |

Please refer to [leaderboard](https://www.tanksandtemples.org/details/691/).

