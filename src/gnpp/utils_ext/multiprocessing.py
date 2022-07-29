import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool, Queue, Process

def parallel_worker(f, part_args, kwargs, index):
    """ f每次接收part_args中一个item作为参数，然后unpacking
    """
    result = []
    if index == 0:
        pbar = tqdm(total=len(part_args), desc='1st process')
    
    for arg in part_args:
        result.append(f(arg, **kwargs))
        if index == 0:
            pbar.update(1)
    
    if index == 0:
        pbar.close()
    return result


class MultiProcessor(object):
    """Encapsulation of multiprocessing
    """
    def __init__(self, pool_size=None):
        if pool_size is None:
            pool_size = multiprocessing.cpu_count() - 2
        else:
            assert pool_size >= 1 and pool_size <= multiprocessing.cpu_count(), "pool size must be in [1, %d]" % multiprocessing.cpu_count()

        self.pool_size = pool_size
    
    def run(self, f, args, kwargs):
        """
        Args:
            args: [arg0, arg1, arg2, ...]
            kwargs: {'a': 0, 'b': 1, ...}
        
        Return:
            [f(arg0, **kwargs), f(arg1, **kwargs), ...]
        
        args is iterable, kwargs is shared.

        An example:
        mp = MultiProcessor(4)
        def func(a, b, c):
            return a + b + c
        a_list = [1, 2, 3, 4, 5]
        b = 1.1
        c = 2.2
        mp.run(func, a_list, {'b': 1.1, 'c': 2.2})

        >>>>>>
        [4.3, 5.3, 6.3, 7.3, 8.3]

        """
       
        batch_size = len(args) // self.pool_size
        indexes = [0, ]
        for i in range(self.pool_size - 1):
            indexes.append(indexes[-1] + batch_size)
        indexes.append(len(args))

        with Pool(processes=self.pool_size) as pool:
            jobs = []
            for i in range(self.pool_size):
                start = indexes[i]
                end = indexes[i + 1]
                part_args = args[start: end]
                
                jobs.append( pool.apply_async(parallel_worker, args=[f, part_args, kwargs, i] ) ) # 添加进程

            for job in jobs:
                job.wait()

        results = []
        for i, j in enumerate(jobs):
            results.extend( j.get() ) # 这里不对结果做特殊处理，只是append，结果的进一步处理， 放在具体的程序里面做。因为不同的实例，其返回值的形式可能不一样。

        return results