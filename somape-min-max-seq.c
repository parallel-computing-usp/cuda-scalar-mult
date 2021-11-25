#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc,char **argv)
{
    //Declara as matrizes
    int *A,*B;
    //Declara as variáveis de índice
    int i,j,dim;
    //Declara o acumulador para o produto escalar global
    int somape, minimo, maximo;

    //Declara um vetor para os produtos escalares locais
    int *prod_escalar;

    FILE *inputfile;     // handler para o arquivo de entrada
    char *inputfilename;  // nome do arquivo de entrada

   
    if (argc < 2)
    {
		printf("Nr errado de argumentos. Execute passando <arq_entrada> como argumento. \n");
		exit(-1);
    }

    inputfilename = (char*) malloc (256*sizeof(char));
    strcpy(inputfilename,argv[1]);

	printf("inputfilename=%s\n", inputfilename);
	fflush(0);

    if ((inputfile=fopen(inputfilename,"r")) == 0)
    {
		printf("Mestre: Erro ao abrir arquivo de entrada %s. Saindo. \n", inputfilename);
		exit(-1);
    }
    
    fscanf(inputfile, "%d\n", &dim); //Lê a dimensão das matrizes
    
    //Aloca as matrizes
    A=(int *)malloc(dim *dim * sizeof(int));
    B=(int *)malloc(dim *dim * sizeof(int)); 

    //Aloca um vetor para armazenar os produtos escalares de cada linha
    prod_escalar=(int *)malloc(dim * sizeof(int));

     //Lê a matriz A
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
            fscanf(inputfile, "%d ",&(A[i*dim+j]));
        }
    }

    //Lê a matriz B
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
            fscanf(inputfile, "%d ",&(B[i*dim+j])); 
        }
    }
	
	// fecha o arquivo de entrada
	fclose(inputfile);

    //Calcula o produto escalar de cada linha
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
            prod_escalar[j]+=A[i*dim+j]*B[i*dim+j];
        }
    }

    //Acumula os produtos das linhas (faz o produto escalar global)
	somape=0;
    for(i=0;i<dim;i++)
	{
        somape+=prod_escalar[i];
    }

    //encontra o mínimo da matriz A
	minimo = A[0];
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
			if (A[i*dim+j]< minimo) 
			{
				minimo = A[i*dim+j];
			}
        }
    }
    //encontra o mínimo da matriz B
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
			if (B[i*dim+j]< minimo) 
			{
				minimo = B[i*dim+j];
			}
        }
    }

   //encontra o máximo da matriz A
	maximo = A[0];
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
			if (A[i*dim+j] > maximo) 
			{
				maximo = A[i*dim+j];
			}
        }
    }
    //encontra o máximo da matriz B
    for(i=0;i<dim;i++)
	{
        for(j=0;j<dim;j++)
		{
			if (B[i*dim+j] > maximo) 
			{
				maximo = B[i*dim+j];
			}
        }
    }

    //Imprime o resultado
    printf("%d %d %d\n",somape, minimo, maximo); 

    //Libera as matrizes
    free(A);
    free(B);

    //Libera o vetor
    free(prod_escalar);
}
