# Gpu_dot
CUDA By Example study  
本代码为CUDA By Example书中关于点积运算的代码，自己仿写，用于自学  
### 学习要点：
1. 共享内存缓冲区：cache  
由于进行数组其中的值相加时，要采用归约(Reduction)操作，会重复对数组内数据进行读写，所以设置操作的数据常驻缓存，这样可以进一步提高代码的运行效率，并且同一个线程块内的所有线程共享该内存区域。**关键字：`__shared__`**  
2. 同步函数：`_syncthreads()`  
注意同步函数每个线程都要执行到，所以尽量避免放入判断语句中。使用同步函数可以避免对共享缓存读写的冲突。  
3. 归约：对一个输入的数组执行某种计算，然后产生更小的结果数组，这个过程称为归约
这里采用的思想时将cache数组中前一半与后一半相加，每次迭代数值数量下降一半。
