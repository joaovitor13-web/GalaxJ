#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, float* massesGPU, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_particles)
    {
        float3 acc = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < n_particles; j++)
        {
            if (i != j)
            {
                float3 diff;
                diff.x = positionsGPU[j].x - positionsGPU[i].x;
                diff.y = positionsGPU[j].y - positionsGPU[i].y;
                diff.z = positionsGPU[j].z - positionsGPU[i].z;

                float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if (dij < EPS)
                {
                    dij = 10.0f;
                }
                else
                {
                    dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
                }

                acc.x += diff.x * dij * massesGPU[j];
                acc.y += diff.y * dij * massesGPU[j];
                acc.z += diff.z * dij * massesGPU[j];
            }
        }

        accelerationsGPU[i] = acc;
    }
}

__global__ void maj_pos(float3 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_particles)
    {
        velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
        velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
        velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

        positionsGPU[i].x += velocitiesGPU[i].x * DIFF_T;
        positionsGPU[i].y += velocitiesGPU[i].y * DIFF_T;
        positionsGPU[i].z += velocitiesGPU[i].z * DIFF_T;
    }
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
    int nthreads = 128;
    int nblocks = (n_particles + (nthreads - 1)) / nthreads;

    compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
    cudaDeviceSynchronize();  // Make sure accelerations are computed before updating positions

    maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
    cudaDeviceSynchronize();  // Ensure position updates are completed
}

#endif // GALAX_MODEL_GPU
