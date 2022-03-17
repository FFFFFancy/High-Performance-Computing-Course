#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

long thread_num;
long num_in_area = 0;
pthread_mutex_t mutex;

double function(double x)
{
    double y = x * x;
    return y;
}

void *thread()
{
    long mu_n = 0;
    srand((unsigned)time(0));
    for(int i=0;i<thread_num;i++)
    {
        double random_x = 0+1.0*rand()/RAND_MAX *(1-0);
        double random_y = 0+1.0*rand()/RAND_MAX *(1-0);
        if(random_y<=function(random_x))
            mu_n++;
    }

    pthread_mutex_lock(&mutex);
	num_in_area += mu_n;
	pthread_mutex_unlock(&mutex);
    return NULL;
}

int main()
{
	long number_of_tosses = 1000000000;
	int thread_count = 10000;
	thread_num = number_of_tosses/thread_count;
    pthread_mutex_init(&mutex, NULL);

	srand((unsigned)time(0));
	pthread_t *thread_handles = malloc(thread_count * sizeof(pthread_t));
    printf("creat threads ...\n");
	for (int i = 0; i < thread_count; i++) {
		pthread_create(&thread_handles[i], NULL, thread, (void *) NULL);
	}
    printf("work is over!\n");
	for (int i = 0; i < thread_count; i++) {
		pthread_join(thread_handles[i], NULL);
	}
	pthread_mutex_destroy(&mutex);
	free(thread_handles);
	printf("Monte Carlo Estimates : %f\n", (double)num_in_area/(double)number_of_tosses);
    printf("Real value of integral : %lf\n",1.0/3);
	return 0;
}