#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void prod_linha(int *d_A, int *d_B, int *d_somape, int dim)
{
    printf("somape brefore: %d\n", *d_somape);
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (lin >= dim || col >= dim) return;
    
    printf("Thread[%d] [%d] multiplica %d * %d\n", lin, col, d_A[lin*dim+col], d_B[lin*dim+col]);
    
    atomicAdd(d_somape, d_A[lin*dim+col]*d_B[lin*dim+col]);
    printf("somape: %d\n", *d_somape);
}



int main(int argc,char **argv)
{
    // //Declara as matrizes do host
    int *h_A,*h_B, h_somape=0;
    // //Declara as matrizes do device
    int *d_A,*d_B, *d_somape;
    // //Declara as variáveis de índice
    int i,j,dim;
    //Declara o acumulador para o produto escalar global
    // int minimo, maximo;


    //Declaração da variável do tipo cudaStream_t
    cudaStream_t stream1;

    //Criação das streams
    cudaStreamCreate(&stream1);
    
    //Lê a dimensão da matriz 
    fscanf(stdin,"%d\n",&dim);
   

    //Aloca as matrizes em memória pinada no host
    cudaHostAlloc((void**)&h_A, dim*dim*(sizeof(int)), cudaHostAllocDefault); 
    cudaHostAlloc((void**)&h_B, dim*dim*(sizeof(int)), cudaHostAllocDefault);

    //Aloca as matrizes no device
    cudaMalloc((void**)&d_A, dim*dim*(sizeof(int))); 
    cudaMalloc((void**)&d_B, dim*dim*(sizeof(int))); 
    cudaMalloc((void**)&d_somape, sizeof(int));
   
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
    
    // Copia de maneira assíncrona a matriz A para o device, dentro da stream 1 
    cudaMemcpyAsync(d_A,h_A,dim*dim*(sizeof(int)),cudaMemcpyHostToDevice,stream1);
    // Copia de maneira assíncrona a matriz B para o device, dentro da stream 1 
    cudaMemcpyAsync(d_B,h_B,dim*dim*(sizeof(int)),cudaMemcpyHostToDevice,stream1);
    // Copia de maneira assíncrona somape=0 para o device, dentro da stream 1 
    cudaMemcpyAsync(d_somape,&h_somape,sizeof(int),cudaMemcpyHostToDevice,stream1);
   
    
    cudaStreamSynchronize(stream1);
    
    //Define a quantidade de threads por bloco
    unsigned int grid_rows = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    prod_linha <<<dimGrid,dimBlock, 0, stream1>>> (d_A,d_B,d_somape, dim);
    cudaStreamSynchronize(stream1);
    
    cudaMemcpyAsync(&h_somape, d_somape, sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    
    printf("%d\n",h_somape); 
//     //Calcula o produto escalar de cada linha
//     for(i=0;i<dim;i++)
// 	{
//         for(j=0;j<dim;j++)
// 		{
//             prod_escalar[j]+=A[i*dim+j]*B[i*dim+j];
//         }
//     }

//     //Acumula os produtos das linhas (faz o produto escalar global)
// 	somape=0;
//     for(i=0;i<dim;i++)
// 	{
//         somape+=prod_escalar[i];
//     }

//     //encontra o mínimo da matriz A
// 	minimo = A[0];
//     for(i=0;i<dim;i++)
// 	{
//         for(j=0;j<dim;j++)
// 		{
// 			if (A[i*dim+j]< minimo) 
// 			{
// 				minimo = A[i*dim+j];
// 			}
//         }
//     }
//     //encontra o mínimo da matriz B
//     for(i=0;i<dim;i++)
// 	{
//         for(j=0;j<dim;j++)
// 		{
// 			if (B[i*dim+j]< minimo) 
// 			{
// 				minimo = B[i*dim+j];
// 			}
//         }
//     }

//    //encontra o máximo da matriz A
// 	maximo = A[0];
//     for(i=0;i<dim;i++)
// 	{
//         for(j=0;j<dim;j++)
// 		{
// 			if (A[i*dim+j] > maximo) 
// 			{
// 				maximo = A[i*dim+j];
// 			}
//         }
//     }
//     //encontra o máximo da matriz B
//     for(i=0;i<dim;i++)
// 	{
//         for(j=0;j<dim;j++)
// 		{
// 			if (B[i*dim+j] > maximo) 
// 			{
// 				maximo = B[i*dim+j];
// 			}
//         }
//     }

    //Imprime o resultado
    // printf("%d %d %d\n",somape, minimo, maximo); 

    //Libera as matrizes
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    
    // //Libera o vetor
    // free(prod_escalar);
}
