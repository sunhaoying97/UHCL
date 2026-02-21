import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging

def get_a_var(obj): #从输入对象中获取一个PyTorch张量。
    if isinstance(obj, torch.Tensor):
        return obj
    ## 如果对象是一个列表或元组，那么对列表或元组中的每个元素递归调用这个函数
    # 并返回第一个找到的PyTorch张量
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    # # 如果对象是一个字典，那么对字典中的每个值递归调用这个函数
    # 并返回第一个找到的PyTorch张量
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids): #在多个设备上并行地应用一个函数到一个模型
    modules = nn.parallel.replicate(model, device_ids) ## 在指定的设备上复制模型
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled) ## 设置梯度是否可用
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)): ## 如果输入不是列表或元组，那么将其转换为元组
                    input = (input,)
                output = fct(module, *input)## 在指定的设备上应用函数到模型
            with lock:
                results[i] = output ## 将结果保存到字典中
        except Exception:# # 如果出现异常，那么将异常信息保存到字典中
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        ## 如果有多个模块，那么创建一个线程池，并在每个线程中运行_worker函数
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:## 如果只有一个模块，那么直接运行_worker函数
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):## 如果结果是一个异常，那么抛出这个异常
            output.reraise()
        outputs.append(output)## 将结果添加到输出列表中
    return outputs

def get_logger(filename=None):#获取一个日志记录器
    logger = logging.getLogger('logger')#名为'logger'
    logger.setLevel(logging.DEBUG)## 设置日志记录器的级别为DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)## 设置基本的日志配置
    if filename is not None:## 如果提供了文件名，那么创建一个文件处理器，并将其添加到日志记录器中
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger