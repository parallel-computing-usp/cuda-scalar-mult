#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void prod_linha(int *d_A, int *d_B, int *d_somape, int dim)
{
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (lin >= dim || col >= dim) return;
    
    // Para fins de debug
    // printf("Thread[%d][%d] multiplica %d * %d\n", lin, col, d_A[lin*dim+col], d_B[lin*dim+col]);
    atomicAdd(d_somape, d_A[lin*dim+col]*d_B[lin*dim+col]);
}

__global__ void minimo_maximo(int *d_Matrix, int *d_minimo, int *d_maximo, int dim)
{
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (lin >= dim || col >= dim) return;
    // Para fins de debug
    // printf("Thread[%d][%d] = %d, mínimo atual: %d, máximo atual: %d\n", lin, col, d_Matrix[lin*dim+col], *d_minimo, *d_maximo);
    atomicMax(d_maximo, d_Matrix[lin*dim+col]);
    __syncthreads();
    *d_minimo = *d_maximo;
    atomicMin(d_minimo, d_Matrix[lin*dim+col]);
}


int main(int argc,char **argv)
{
    // //Declara as matrizes do host
    int *h_A,*h_B;
    // //Declara as matrizes do device
    int *d_A,*d_B;
    //Declara as variáveis de resultados para o host
    int h_minimo = 0, h_maximo = 0, h_somape =0;
    //Declara as variáveis de resultados para o device
    int *d_minimo, *d_maximo, *d_somape;
    // //Declara as variáveis de índice
    int i,j,dim;
   


    //Declaração da variável do tipo cudaStream_t
    cudaStream_t stream1, stream2, stream3;

    //Criação das streams
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    //Lê a dimensão da matriz 
    fscanf(stdin,"%d\n",&dim);
   

    //Aloca as matrizes em memória pinada no host
    cudaHostAlloc((void**)&h_A, dim*dim*(sizeof(int)), cudaHostAllocDefault); 
    cudaHostAlloc((void**)&h_B, dim*dim*(sizeof(int)), cudaHostAllocDefault);

    //Aloca as matrizes e as variáveis de output no device
    cudaMalloc((void**)&d_A, dim*dim*(sizeof(int))); 
    cudaMalloc((void**)&d_B, dim*dim*(sizeof(int))); 
    cudaMalloc((void**)&d_somape, sizeof(int));
    cudaMalloc((void**)&d_minimo, sizeof(int));
    cudaMalloc((void**)&d_maximo, sizeof(int));
   
    //Lê a matriz A do host
    for(i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            fscanf(stdin, "%d ",&(h_A[i*dim+j]));
        }
    }
    
    //Lê a matriz B do host
    for(i=0;i<dim;i++)
	{ for(j=0;j<dim;j++){
            fscanf(stdin, "%d ",&(h_B[i*dim+j])); 
        }
    }

    //Define a quantidade de threads por bloco
    unsigned int grid_rows = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Copia de maneira assíncrona a matriz A para o device, dentro da stream 1 
    cudaMemcpyAsync(d_A,h_A,dim*dim*(sizeof(int)),cudaMemcpyHostToDevice,stream1);
    // Copia de maneira assíncrona a matriz B para o device, dentro da stream 1 
    cudaMemcpyAsync(d_B,h_B,dim*dim*(sizeof(int)),cudaMemcpyHostToDevice,stream1);
    // Copia de maneira assíncrona somape=0 para o device, dentro da stream 1 
    cudaMemcpyAsync(d_somape,&h_somape,sizeof(int),cudaMemcpyHostToDevice,stream1);
   
    // Copia de maneira assíncrona minimo=0 para o device, dentro da stream 2 
    cudaMemcpyAsync(d_minimo,&h_minimo,sizeof(int),cudaMemcpyHostToDevice,stream2);
    // Copia de maneira assíncrona maximo=0 para o device, dentro da stream 2 
    cudaMemcpyAsync(d_maximo,&h_maximo,sizeof(int),cudaMemcpyHostToDevice,stream2);

     // Copia de maneira assíncrona minimo=0 para o device, dentro da stream 3 
    cudaMemcpyAsync(d_minimo,&h_minimo,sizeof(int),cudaMemcpyHostToDevice,stream3);
    // Copia de maneira assíncrona maximo=0 para o device, dentro da stream 3 
    cudaMemcpyAsync(d_maximo,&h_maximo,sizeof(int),cudaMemcpyHostToDevice,stream3);
   
    
    cudaStreamSynchronize(stream1);
    prod_linha <<<dimGrid,dimBlock, 0, stream1>>> (d_A,d_B,d_somape, dim);
    cudaMemcpyAsync(&h_somape, d_somape, sizeof(int), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream2);
    minimo_maximo<<<dimGrid, dimBlock, 0, stream2>>> (d_A, d_minimo, d_maximo, dim);
    cudaMemcpyAsync(&h_minimo, d_minimo, sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(&h_maximo, d_maximo, sizeof(int), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream2);
    printf("Mínimo da Matriz A: %d\n",h_minimo); 
    printf("Máximo da Matriz A: %d\n",h_maximo); 

    cudaStreamSynchronize(stream3);
    minimo_maximo<<<dimGrid, dimBlock, 0, stream2>>> (d_B, d_minimo, d_maximo, dim);
    cudaMemcpyAsync(&h_minimo, d_minimo, sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(&h_maximo, d_maximo, sizeof(int), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream3);
    printf("Mínimo da Matriz B: %d\n",h_minimo); 
    printf("Máximo da Matriz B: %d\n",h_maximo);

    cudaStreamSynchronize(stream1);
    printf("Produto escalar: %d\n",h_somape); 

    

    
 

    //Libera as matrizes
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(&h_somape);
    cudaFreeHost(&h_minimo);
    cudaFreeHost(&h_maximo);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_somape);
    cudaFree(d_minimo);
    cudaFree(d_maximo);

    
    // //Libera o vetor
    // free(prod_escalar);
}
