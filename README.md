# cupidshuffle
Down, down, do your dance, do your dance

![](memes/cupid-shuffle.gif)

# Repo for the Down, down, do your dance, do your dance: CupidShuffle paper
paper located in memes folder

# Setup

The following works on Ubuntu 20.04 : 

    conda create --name cupidshuffle python=3.7
    conda activate cupidshuffle

per https://pytorch.org/get-started/locally/

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

and lastly

    python -m pip install numpy tensorboard einops timm apache-tvm chardet

Now you should be able to run the following


# Training

```bash
python3 train.py
```

# Compiling

```bash
python3 compile.py
```
or

```python
from model import CupidShuffle
from compile import TVMCompiler

net = CupidShuffle(start_channels=28, token_dim=28, repeats=[1,4,1])
# load our weights
net.load_state_dict(torch.load("weights/cupidshuffle.pth"))
# load our compiler
compiler = TVMCompiler(
    height = 224, 
    width = 224, 
    input_name = "input0",
    dtype = "float32",
    target = 'llvm', # for arm llvm -device=arm_cpu -mtriple=aarch64-linux-gnu
    save_path = 'cupidshufflenet_tvm',
    log_filename = 'cupidshufflenet_tvm.log',
    graph_opt_sch_file  = 'cupidshufflenet_tvm_graph_opt.log',
    tuner =  'xgb',
    n_trial =  10,
    early_stopping =  None,
    use_transfer_learning =  True,
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(
          number=20, repeat=3, timeout=4, 
          min_repeat_ms=150, enable_cpu_cache_flush=True
          )
        ),
    )
# export our model
compiler.export(net, True)
```
# Usage python

```python
from model import CupidShuffle

net = CupidShuffle(start_channels=28, token_dim=28, repeats=[1, 4, 1])

output = net(image)
```

# Usage c++

```c++
//create model pointer 
std::unique_ptr<CupidShuffle> cupidshuffle;
cupidshuffle.reset(new CupidShuffle("/location/to/ccp.json"));

//fill image from pointcoud
cv::Mat cv_image (cloud_->height, cloud_->width, CV_8UC3);
cv::Mat cv_depth_image (cloud_->height, cloud_->width, CV_32FC1);
for (int i=0;i<cloud_->height;i++)
{
    for (int j=0;j<cloud_->width;j++)
    {
    cv_image.at<cv::Vec3b>(i,j)[2] = cloud_->at(j,i).r;
    cv_image.at<cv::Vec3b>(i,j)[1] = cloud_->at(j,i).g;
    cv_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).b;
    cv_depth_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).z;
    }
}

// get your output and get funky
MatF output = cupidshuffle->forward(cv_image);
```
