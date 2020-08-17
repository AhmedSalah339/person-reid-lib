# person-reid-lib

The pytorch-based lightweight library of person re-identification.

#### Config

---

Version
```
python 3.6 or 3.7
pytorch >= 0.4
```


Install

```
pip install numpy h5py lmdb
pip install visdom  # Optional. If you don't need a web page visualization, don't install it.
```
Install [pytorch and torchvision](https://pytorch.org/)


Indicates the folder of the original files and where the unzipped file is placed.
```
# person-reid-lib/lib/utils/manager.py
self._device_dict = xxxx
```


#### Optical Flow

---

Install opencv
```
pip install opencv-contrib-python    # version 3.4.2.17
```

Config
```
# person-reid-lib/lib/dataset/utils.py
DataStoreManager.store_optical_flow = True  # if you want to use optical flow, enable it.

# person-reid-lib/tasks/taskname/solver.py
Solver.use_flow = True
```

#### How to run:

---

```
# image-dataset
cd person-reid-lib_folder
sh script/server_0.sh

# video-dataset

cd person-reid-lib_folder
sh script/task_video.sh
```

#### Dataset

---

Image: VIPeR, Market1501, CUHK03, CUHK01, DukeMTMCreID, GRID,

Video : iLIDS-VID, PRID-2011, LPW, MARS, DukeMTMC-VideoReID

#### Updates

---
**2018.12.29**  The code of [Spatial and Temporal Mutual Promotion for Video-based Person Re-identification](https://arxiv.org/abs/1812.10305) is available.

**2018.12.26**  The initial version is available.

**2018.11.19**  The code for *lib* has been released.
# Tutorial
-[Choose to train or test](#choose-to-train-or-test)
-[Choose data set](#choose-data-set)
-[Control the number of task repeating](#control-the-number-of-task-repeating)
-[Saving and reusing the model](#saving-and-reusing-the-model)
-[Change epochs num](#change-epochs-num)
-[Make new dataset](#)
-[Gets a single prediction](#gets-a-single-prediction)
The main which the program starts with is on train_test.py:
person-reid-lib/tasks/task_video/train_test.py /

It has two main jobs:

### 1- initialising the data manager 
### Choose to train or test
```manager = Manager(cur_dir, seed=None, mode='Train')```

The mode has two possiple values 'Train' and 'Test' 

Following that it sets the data set used
### Choose data set
``` manager.set_dataset(0)```

where 
```
['iLIDS-VID', 'PRID-2011', 'LPW', 'MARS', 'VIPeR', 'Market1501', 'CUHK03', 'CUHK01', 'DukeMTMCreID', 'GRID', 'DukeMTMC-VideoReID']
           0            1         2      3        4          5           6         7             8           9             10
```
### 2-Creates the solver and running it

The solver is the class which trains and tests the model on the required data set
#### Control the number of task repeating
```
repeat_times = 10
for task_i in range(repeat_times):
        manager.split_id = int(task_i) 
        task = Solver(manager)
```
by default it makes 10 train test tasks which in each it uses a new random split for the training and test every epoch

### Solver class

person-reid-lib/tasks/task_video/solver.py /

The solver has important function that sets options for saving and reusing the model and for saving the search results and also initialises the net client and the model client which deals with the model.
#### Saving and reusing the model
```
def init_options(self):
        # ------option------
        self.use_flow = False
        self.save_model = False
        self.reuse_model = False
        self.store_search_result = False
        self.net_client = NetClient
        self.model_client = ModelClient
```

The solver class inherits from TaskSolverBase which has the real functionality lib/network/solver_factory/solverbase.py /

The TaskSolverBase does some important things after initialisation

# 1- sets the data manager
# 2- sets the evaluator
---
The run function in this class is what is called to start the train-test task

```
    def run(self):
        self.network = NetManager(nClass=self.Data.dataset.train_person_num,
                                  nCam=self.Data.dataset.train_cam_num,
                                  net_client=self.net_client,
                                  model_client=self.model_client,
                                  use_flow=self.use_flow,
                                  task_dir=self.manager.task_dir,
                                  raw_model_dir=self.manager.device['Model'],
                                  is_image_dataset=self.Data.dataset.is_image_dataset,
                                  recorder=self.recorder)
        if self.task_mode == 'Train':
            self.train_test()
        else:
            self.test()
        if self.store_search_result:
            self.evaluator.store_example()
```

it basically creates the network object which is basically sets some variables and returns a netclient object that deals with the model and its forward, back word and evaluation

### Note: The number of cameras and persons is retrieved from the datamanager object (Data) which takes the dataset name  and gets info about it


## train_test

The train_test function 
```
        while train_flag:
            self.train(network)
            train_flag, test_flag = self.manager.check_epoch(self.epoch)

            if self.epoch % self.display_step == 0:
                network.display()

            if test_flag:
                cmc, mAP = self.do_eval(network, self.test_batch_size)

                network.mode = 'Train'
                self.Data.set_transform(network.model.get_transform())

                self.perf_box[str(self.epoch)] = {'cmc': cmc, 'mAP': mAP}
                if self.save_model:
                    network.save(cmc[0])
```
basically this is the main train test task which manages the training epochs, when to stop and start testing and displaying the epochs info

the epochs num is advancing in the train function and the chech epoch tells when to stop training and start testing.
### Change epochs num

### Note: to control epochs num in the folder "self.task_dir / 'test_epoch_id.txt'" the epoch size is the third number 
### (in the code person-reid-lib/lib/utils/manager.py / function check epochs "epoch_size = info_list[2]")

train and test functions are basic and easy and dont have much to do with them but the network object is very important.

It is a NetClient object (person-reid-lib/tasks/task_video/architecture.py) contains forward_backward method and eval method which takes a batch of the data and do training epoch or a testing epoch 

(you can find these methods in NetBase class person-reid-lib/lib/network/model_factory/netbase.py /).

The code for the evaluation in the solver base is 
```
    def eval(self, network, batch_size):
        self.evaluator.reset(self.Data.dataset, distance_func=network.model.distance_func)
        self.evaluator.set_feature_buffer(network.model.fea_process_func)
        self.logger.info('Test Begin')
        network.mode = 'Test'
        self.Data.set_transform(network.model.get_transform())
        dataloader = self.Data.get_test(batch_size)
        for i_batch, batch_data in enumerate(dataloader):
            feature = network.eval(batch_data)
            self.evaluator.count(feature)

        result = self.evaluator.final_result()
        return result

    def do_eval(self, network, batch_size):
        result = self.eval(network, batch_size)
        cmc = result[0]
        mAP = result[1]
        cmc_rank = [float(cmc[0]), float(cmc[4]), float(cmc[9]), float(cmc[19])]
        return cmc_rank, mAP
```
### Get a single response on some images

Basically there are two steps to do so,

#### * Get the features from the model

In which you prepare the data using the data managaer object (Data in this case) and then get the features by using network.eval()

#### * Calculate the distances and take the decision

Which is basically implemented in the evaluator 

It has rules for every dataset to evaluate with

```
self._eval_factory = {'PRID-2011': EvalStandard,
                              'iLIDS-VID': EvalStandard,
                              'MARS': EvalMARS,
                              'LPW': EvalStandard,
                              'CUHK01': EvalStandard,
                              'CUHK03': EvalStandard,
                              'DukeMTMCreID': EvalStandard,
                              'GRID': EvalStandard,
                              'Market1501': EvalStandard,
                              'VIPeR': EvalStandard,
                              'DukeMTMC-VideoReID': EvalStandard}
```
Only Mars dataset is evaluated differently

In eval_base (person-reid-lib/lib/evaluation/evaluation_rule/eval_base.py)

```
    def _feature_distance(self, feaMat):
        probe_feature = torch.index_select(feaMat, dim=0, index=torch.from_numpy(self.probe_index).long().cuda())
        gallery_feature = torch.index_select(feaMat, dim=0, index=torch.from_numpy(self.gallery_index).long().cuda())

        idx = 0
        while idx + self.probe_dst_max < self.probe_num:
            tmp_probe_fea = probe_feature[idx:idx+self.probe_dst_max]
            dst_pg = self._feature_distance_mini(tmp_probe_fea, gallery_feature)
            self.distMat[idx:idx+self.probe_dst_max] += dst_pg
            idx += self.probe_dst_max
        tmp_probe_fea = probe_feature[idx:self.probe_num]
        dst_pg = self._feature_distance_mini(tmp_probe_fea, gallery_feature)
        self.distMat[idx:self.probe_num] += dst_pg

        for i_p, p in enumerate(self.probe_index):
            for i_g, g in enumerate(self.gallery_index):
                if self.test_info[p, 0] != self.test_info[g, 0]:
                    self.avgDiff = self.avgDiff + self.distMat[i_p, i_g]
                    self.avgDiffCount = self.avgDiffCount + 1
                elif p != g:
                    self.avgSame = self.avgSame + self.distMat[i_p, i_g]
                    self.avgSameCount = self.avgSameCount + 1
```

Here the code matche the features of all the probe folder and the gallery folder to thier indices and  and then compares betwwen them and uses the test info (the data manager prepares it) to know if the difference in the features is big or not 

this code can be modified to get one feature using the network class and then compares it with the gallery images and get the minimum difference.

#### Related person ReID projects:

---

[deep person reid](https://github.com/KaiyangZhou/deep-person-reid)

[MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation)
