#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

double a, b, c;
double delta,radical;
pthread_mutex_t mutex;
pthread_cond_t cond1,cond2;

void *work_1(void *rank)
{
    pthread_mutex_lock(&mutex);
    delta = b * b - 4 * a * c;
    if (delta < 0)
    {
        printf("The equation has no solution.\n");
        exit(0);
    }

    pthread_cond_signal(&cond1);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void *work_2(void *rank)
{
    pthread_mutex_lock(&mutex);
    while (pthread_cond_wait(&cond1, &mutex) != 0);
    radical = sqrt(delta);
    pthread_cond_signal(&cond2);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char *argv[])
{
    printf("Quadratic equations of one variable: ax^2+bx+c=0\n");
    printf("Please enter a, b and c: ");
    scanf("%lf %lf %lf", &a, &b, &c);

    double x1,x2;
    pthread_mutex_init(&mutex,0);
	pthread_cond_init(&cond1,NULL);
    pthread_cond_init(&cond2,NULL);

    pthread_t thread1,thread2;
    pthread_create(&thread1, NULL, work_1,(void *)0);
	pthread_create(&thread2, NULL, work_2,(void *)1);

    pthread_mutex_lock(&mutex);
    while (pthread_cond_wait(&cond2, &mutex) != 0);
    pthread_mutex_unlock(&mutex);

    pthread_cond_destroy(&cond1);
    pthread_cond_destroy(&cond2);
	pthread_mutex_destroy(&mutex);

    pthread_join(thread1,NULL);
	pthread_join(thread2,NULL);

    x1 = (-b+radical)/(2*a);
    x2 = (-b-radical)/(2*a);
    if (x1 == x2)
        printf("Solution: x1 = x2 = %f\n", x1); 
    else
        printf("Solution: x1 = %f , x2 = %f\n", x1, x2);

    return 0;
}