#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "utils.h"

int main(void)
{
    // context init ok
    checkCudaErrors(cudaFree(0));

    // generate 32M random numbers serially (2^20 = 1M)
    // thrust::host_vector<int> h_vec(32 << 20);
    // generate 1M random numbers serially (2^20 = 1M)
    thrust::host_vector<int> h_vec(1 << 20);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // print d_vec, size = d_vec.size()
    for(int i = 0; i < 50; i++)
        std::cout << "h_dev[" << i << "] = " << h_vec[i] << std::endl;

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

     // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    // print d_vec, size = d_vec.size()
    for(int i = 0; i < 50; i++)
        std::cout << "h_dev[" << i << "] = " << h_vec[i] << std::endl;

  return 0;
}