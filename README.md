# verification_code
python+sklearn识别字母数字验证码



## 项目结构
```
|--image_after_split                        // 切割后的图片
|--image_raw                                // 原图片
|--model_data                               // 训练后的模型存放
|  `--letter.pkl
|--test_img                                 // 验证图片
|--train_img                                // 用于训练的图片存放
|  |--capital                               // 大写字母
|  |--lowercase                             // 小写字母
|  |--num                                   // 数字
|─utils
|  `--deal_image.py                         // 图片处理程序                      
|-- requirements.txt                        // 项目依赖
|-- split_image.py                          // 图片预处理及切割
|-- test_model.py                           // 测试模型的识别效果
|-- train.py                                // 开始训练
```