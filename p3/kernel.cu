/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Básico 3
*
* Alumno: Rodrigo Pascual Arnaiz
* Fecha: 06/10/2022
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h> 
#include <device_launch_parameters.h>
///////////////////////////////////////////////////////////////////////////
// defines
#define BLOQUE_TAM 10

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host

/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
* es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: cudaDeviceProp -> retorna el onjeto que tiene todas las 
* propiedades del dispositivo CUDA
*/
__host__ cudaDeviceProp propiedadesDispositivo( int id_dispositivo )
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, id_dispositivo);
	// calculo del numero de cores (SP)
	int cuda_cores = 0;
	int multi_processor_count = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	switch (major)
	{
	case 1:
		//TESLA
		cuda_cores = 8;
		break;
	case 2:
		//FERMI
		if (minor == 0)
			cuda_cores = 32;
		else
			cuda_cores = 48;
		break;
	case 3:
		//KEPLER
		cuda_cores = 192;
		break;
	case 5:
		//MAXWELL
		cuda_cores = 128;
		break;
	case 6:
		//PASCAL
		cuda_cores = 64;
		break;
	case 7:
		//VOLTA
		cuda_cores = 64;
		break;
	case 8:
		//AMPERE
		cuda_cores = 128;
		break;
	default:
		//DESCONOCIDA
		cuda_cores = 0;
	}
	if (cuda_cores == 0)
	{
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DISPOSIRIVO %d: %s\n", id_dispositivo, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> N. de MultiProcesadores \t\t: %d \n", multi_processor_count);
	printf("> N. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores,
		multi_processor_count, cuda_cores * multi_processor_count);
	printf("> N. max. de Hilos (por bloque) \t: %d \n",
		deviceProp.maxThreadsPerBlock);
	printf(
		" [Eje x\t->\t%d]\n [Eje y\t->\t%d]\n [Eje z\t->\t%d]\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]
	);
	printf("> N. Max. de Bloques (por eje)\n");
	printf(
		" [Eje x\t->\t%d]\n [Eje y\t->\t%d]\n [Eje z\t->\t%d]\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]
	);

	printf("***************************************************\n");
	return deviceProp;
}



/**
* Funcion: rellenarVectorHst
* Objetivo: Funcion que rellena un array pasado por parametro
* con numero aleatorios del 0 al 9
*
* Param: INT* arr -> Puntero del array a rellenar
* Param: INT size -> Longitud del array
* Return: void
*/
__host__ void rellenarVectorHst(int* arr, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		arr[i] = rand() % 10;
	}
}

/**
* Funcion: invertirVector
* Objetivo: Funcion que da la vuelta a un vector pasado por paramtro
*
* Param: INT* arr -> Puntero del array a invertir
* Param: INT size -> Longitud del array
* Return: void
*/
__host__ void invertirVector(int* arr, int size)
{

	int temporal;
	for (int i = 0, x = size - 1; i < x; i++, x--) {
		temporal = arr[i];
		arr[i] = arr[x];
		arr[x] = temporal;
	}

}


/**
* Funcion: sumarArrays
* Objetivo: Funcion que da la vuelta a un vector pasado por paramtro
*
* Param: INT* primer_array -> Primer puntero del array que se quiere sumar
* Param: INT* segundo_array -> Segundo puntero del array que se quiere sumar
* Param: INT* array_resultado -> Puntero del array que va a contener el resultado
* Param: INT size -> Tamaño de los arrays
* Return: void
*/

__global__ void sumarArrays(int* primer_array, int* segundo_array, int* array_resultado, int size)
{
	int idT = threadIdx.x;
	int idB = blockIdx.x;
	int pos = BLOQUE_TAM * idB + idT;
	if (pos < size)
	{
		array_resultado[pos] = primer_array[pos] + segundo_array[pos];
	}
}

///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Semilla de random aleatoria 
	srand( time( NULL ) );

	// Obetener el dispisivo cuda
	int numero_dispositivos;
	cudaDeviceProp propiedades_dispositivo;
	cudaGetDeviceCount(&numero_dispositivos);

	if ( numero_dispositivos == 0 )
	{
		printf("!!!!!ERROR!!!!!\n");
		printf("Este ordenador no tiene dispositivo de ejecucion CUDA\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;

	}
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", numero_dispositivos);
		for (int id = 0; id < numero_dispositivos; id++)
		{
			propiedades_dispositivo = propiedadesDispositivo(id);
		}
	}

	//************	3. Ejercicio  ************//
	// declaracion de variables
	int* hst_vector1, * hst_vector2, * hst_resultado;
	int* dev_vector1, * dev_vector2, * dev_resultado;


	int numero_bloques; // Número de bloques necesarios
	int numero_elementos;
	bool is_numero_valido = false;
	bool is_cantidad_valida = false;

	do {
		do {
			printf("Introduce el numero de elementos: ");
			is_numero_valido = scanf("%i", &numero_elementos);
			printf("\n");
		} while (!is_numero_valido);


		numero_bloques = ceil( (float)numero_elementos / (float)BLOQUE_TAM );
		printf("Utilizando %i bloques de %i hilos (%i hilos)\n\n", numero_bloques, BLOQUE_TAM, numero_bloques* BLOQUE_TAM);
		if ( numero_bloques < propiedades_dispositivo.maxGridSize[0]  )
		{
			is_cantidad_valida = true;
		}
		else {
			printf("> ERROR: numero maximo de bloques superado! [ %d bloques]\n", propiedades_dispositivo.maxGridSize[0]);
		}

	} while (!is_cantidad_valida);

	printf("> Vector de %d elementos \n", numero_elementos);

	
	// reserva de memoria en el host
	hst_vector1 = (int*)malloc(numero_elementos * sizeof(int));
	hst_vector2 = (int*)malloc(numero_elementos * sizeof(int));
	hst_resultado = (int*)malloc(numero_elementos * sizeof(int));

	// reserva de memoria en el device
	cudaMalloc((void**)&dev_vector1, numero_elementos * sizeof(int));
	cudaMalloc((void**)&dev_vector2, numero_elementos * sizeof(int));
	cudaMalloc((void**)&dev_resultado, numero_elementos * sizeof(int));



	// Rellenamos el vector con la funcion previamente creada 
	rellenarVectorHst( hst_vector1, numero_elementos);

	
	// Copiamos el vector hst 1 al vector hst 2, la ide es invertir la copia en hst 2
	cudaMemcpy( hst_vector2, hst_vector1, numero_elementos * sizeof(int), cudaMemcpyHostToHost );

	// Invertimos el vector y ese mismo vector es el resultado
	invertirVector(hst_vector2, numero_elementos );

	// Copiamos el vextor invertido en  h
	cudaMemcpy( dev_vector2, hst_vector2, numero_elementos * sizeof(int), cudaMemcpyHostToDevice );

	// Copiamos el contenido del vector device 2 al vector host 2
	cudaMemcpy( hst_vector2, dev_vector2, numero_elementos * sizeof(int), cudaMemcpyDeviceToHost);


	// Mostrar vector 1
	printf("VECTOR 1:\n");
	for (int i = 0; i < numero_elementos; i++)
	{
		printf("%i ", hst_vector1[i]);
	}
	printf("\n");
	// Mostrar vector 2
	printf("VECTOR 2:\n");
	for (int i = 0; i < numero_elementos; i++)
	{
		printf("%i ", hst_vector2[i]);
	}
	printf("\n");


	// La suma será realizada en el device aprovechando todos los hilos o threads lanzados sin sobrepasar el máximo permitido.


	// Sumar V1 + V2
	cudaMemcpy(dev_vector1, hst_vector1, numero_elementos * sizeof(int), cudaMemcpyHostToDevice);
	sumarArrays <<<numero_bloques, BLOQUE_TAM >>> (dev_vector1, dev_vector2, dev_resultado, numero_elementos);
	cudaMemcpy(hst_resultado, dev_resultado, numero_elementos * sizeof(int), cudaMemcpyDeviceToHost);

	// Mostrar resultado de la suma
	printf("\nSUMA:\n", numero_elementos);
	for (int i = 0; i < numero_elementos; i++)
	{
		printf("%i ", hst_resultado[i]);
	}
	printf("\n");

	// Salida del programa
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;

}
///////////////////////////////////////////////////////////////////////////