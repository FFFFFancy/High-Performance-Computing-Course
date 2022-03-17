#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

pthread_mutex_t mutex;
int array[1000];
int threads_num;
int global_index = 0;
int sum = 0;

void* add_array ()
{
	pthread_mutex_lock(&mutex);
	sum += array[global_index];
	global_index++;
	pthread_mutex_unlock(&mutex);
}

int main(int argc, char **argv)
{
    clock_t begin, end;
    double time1;
    threads_num=strtol(argv[1], NULL, 10);
    pthread_t thread[threads_num];
    pthread_mutex_init(&mutex, NULL);
    srand((unsigned)time(0));
    for (int i = 0; i < 1000; i++)
    {
        array[i] = (int)rand() % 100;
    }
    // printf("数组元素：");
	// 	for (int i=0;i<1000;i++)
	// 		printf("%d ",array[i]);
	// 	printf("\n");

    begin = clock();

    for (int i=0;i<threads_num;i++){
		for (int t=0;t<1000/threads_num;t++) 
			pthread_create(&thread[i], NULL, add_array, NULL);
	}
	for (int i=0;i<threads_num;i++){
		for (int t=0;t<1000/threads_num;t++) 
			pthread_join(thread[i], NULL);
	}
	pthread_mutex_destroy(&mutex);

    end = clock();
    time1 = (double)(end-begin)/CLOCKS_PER_SEC;;

    printf("sum = %d\n", sum);
    printf("time of array_sum: %f s\n", time1);
}
