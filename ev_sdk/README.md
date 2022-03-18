# SDK封装说明

平台提供了算法自动测试服务，意在模拟算法真实场景下的落地过程，并且为大家提供方便的模型测试工具。因此，您需要按照平台的要求封装SDK，也就是按照平台封装模型推理时的输入输出。您可以选择Python或者C++来封装，为了简便这里建议您使用Python。

**注意：**每个赛题/项目的封装要求可能有所不同，如输入参数、输出数据格式等，请参考对应赛题或者项目的相关说明。

## 1. 支持的接口

### 1.1 Python接口

用户需要按照要求实现函数接口，发起测试时，系统会调用文件`/project/ev_sdk/src/ji.py`，根据具体算法的封装要求，系统会将测试图片逐次送入`process_image`接口。

通常需要实现以下接口：

```python
# 初始化接口
init()
# 处理数据
process_image(handle=None, input_image=None, args=None, **kwargs)
```

**注意：**通常`process_image`即为算法输出内容，其格式必须符合对应赛题/项目的要求。

### 1.2 C/C++接口

用户需要按照要求实现函数接口，并将实现好的程序编译成库存放到`/project/ev_sdk/lib/libji.so`，发起测试时，系统会调用此文件进行测试，根据具体算法的封装要求系统会将测试图片逐次送入`ji_calc_frame`接口，接口说明请参考`/project/ev_sdk/include/ji.h`。

通常需要实现以下接口：

```c++
// 初始化：
int ji_init(int argc, char **argv)
// 创建算法句柄
void* ji_create_predictor(int pdtype)
// 处理数据
int ji_calc_frame(void *predictor, const JI_CV_FRAME *inFrame, const char *args, JI_CV_FRAME *outFrame, JI_EVENT *event)
```

**注意：**参数`event`是输出参数指针，其内容`event.json`即为算法输出内容，其格式必须符合对应赛题/项目的要求。

## 2. 默认示例代码
默认作为示例提供了两部分内容：

1. 提供了一个使用`darknet`框架实现的检测算法，使用C++SDK进行封装；
2. 提供了一份默认的Python封装代码，里面包含基本的封装接口说明，不包含具体的模型推理代码；

### 2.1 C++代码的使用方法

1. 编译和安装`libji.so`：

   ```shell
   mkdir -p /usr/local/ev_sdk/build
   cd /usr/local/ev_sdk/build
   cmake ..
   make install
   ```

   执行完成后， 会生成`/usr/local/ev_sdk/lib/libji.so`库文件，以及测试程序`/usr/local/ev_sdk/bin/test-ji-api`。

2. 使用`test-ji-api`测试`ji_calc_frame`接口，测试单张图片：

   ```shell
   /usr/local/ev_sdk/bin/test-ji-api -f ji_calc_frame -i /usr/local/ev_sdk/data/dog.jpg
   ```

   输出内容样例：

   ```shell
    code: 0
    json: {
    "alert_flag":   1,
    "dogs": [{
            "xmin": 129,
            "ymin": 186,
            "xmax": 369,
            "ymax": 516,
            "confidence":   0.566474,
            "name": "dog"
        }]
   }
   ```

### 2.2 Python代码的使用方法

默认Python代码只提供了简单的接口说明，请根据具体的赛题/项目要求进行实现。

