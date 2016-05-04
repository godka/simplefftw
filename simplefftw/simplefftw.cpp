// simplefftw.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "fftw3.h"  
#include <stdio.h>  
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <iostream>// 复数类型结构体
using namespace std;
void equlization(fftw_complex* data, int width, int height);
int SimpleFFTshift(int width, int height, fftw_complex* in){
	//by rows.
	int half_width = floor(width / 2);
	int half_height = floor(height / 2);

	for (auto i = 0; i < height; i++){
		for (auto j = 0; j < half_width; j++){
			auto pixel = i * width + j;
			auto pixel2 = i * width + j + half_width;
			auto tmpreal = in[pixel][0];auto tmpimage = in[pixel][1];
			in[pixel][0] = in[pixel2][0];in[pixel][1] = in[pixel2][1];
			in[pixel2][0] = tmpreal;in[pixel2][1] = tmpimage;
		}
	}

	for (auto i = 0; i < half_height; i++){
		for (auto j = 0; j < width; j++){
			auto pixel = i * width + j;
			auto pixel2 = (i + half_height) * width + j;
			auto tmpreal = in[pixel][0];auto tmpimage = in[pixel][1];
			in[pixel][0] = in[pixel2][0];in[pixel][1] = in[pixel2][1];
			in[pixel2][0] = tmpreal;in[pixel2][1] = tmpimage;
		}
	}
	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{/* load original image */
	IplImage *img_src = cvLoadImage("x1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (img_src == 0)
	{
		std::cout << "cannot load file" << std::endl;
		return 0;
	}

	/* create new image for FFT & IFFT result */
	IplImage *img_fft = cvCreateImage(cvSize(img_src->width, img_src->height), IPL_DEPTH_8U, 1);
	IplImage *img_ifft = cvCreateImage(cvSize(img_src->width, img_src->height), IPL_DEPTH_8U, 1);

	/* get image properties */
	int width = img_src->width;
	int height = img_src->height;
	int step = img_src->widthStep;
	uchar *img_src_data = (uchar *) img_src->imageData;
	uchar *img_ifft_data = (uchar *) img_ifft->imageData;

	/* initialize arrays for fftw operations */
	fftw_complex *data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * width * height);
	fftw_complex *fft = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * width * height);
	fftw_complex *ifft = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * width * height);

	/* create plans */
	fftw_plan plan_f = fftw_plan_dft_2d(height, width, data_in, fft, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_b = fftw_plan_dft_2d(height, width, fft, ifft, FFTW_BACKWARD, FFTW_ESTIMATE);

	int i, j, k;
	for (i = 0, k = 0; i < height; ++i){
		for (j = 0; j < width; ++j){
			data_in[k][0] = (double) img_src_data[i * step + j];
			data_in[k++][1] = 0.0;
		}
	}

	fftw_execute(plan_f);
	SimpleFFTshift(width, height, fft);

	auto n1 = floor(width / 2);
	auto n2 = floor(height / 2);
	auto d0 = 30;
	for (auto i = 0; i < height; i++){
		for (auto j = 0; j < width; j++){
			//complex<double> _complex(fft[i * width + j][0], fft[i * width + j][1]);
			auto d = pow(j - n2, 2) + pow(i - n1, 2);
			auto h = 1 - exp(-d / (2 * (d0 ^ 2)));
			h = 0.5 + 0.75*h;
			fft[i * width + j][0] *= h;
			fft[i * width + j][1] *= h;
		}
	}
	/* perform IFFT */

	SimpleFFTshift(width, height, fft);
	fftw_execute(plan_b);
	/* normalize IFFT result */
	for (i = 0; i < width * height; ++i)
	{
		ifft[i][0] /= width * height;
		if (ifft[i][0] > 255)ifft[i][0] = 255;
		if (ifft[i][0] < 0)ifft[i][0] = 0;
	}
	equlization(ifft, width, height);
	/* copy IFFT result to img_ifft's data */
	for (i = 0, k = 0; i < height; ++i)
		for (j = 0; j < width; ++j)
			img_ifft_data[i * step + j] = (uchar) ifft[k++][0];

	/* display images */
	cvNamedWindow("original_image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("IFFT", CV_WINDOW_AUTOSIZE);
	cvShowImage("original_image", img_src);
	cvShowImage("IFFT", img_ifft);

	cvWaitKey(0);

	/* free memory */
	cvDestroyWindow("original_image");
	cvDestroyWindow("IFFT");
	cvReleaseImage(&img_src);
	cvReleaseImage(&img_ifft);
	fftw_destroy_plan(plan_f);
	fftw_destroy_plan(plan_b);
	fftw_free(data_in);
	fftw_free(fft);
	fftw_free(ifft);
}
void equlization(fftw_complex* data, int width, int height){
	unsigned int his[256] = {0};
	unsigned int temp = 0;
	unsigned int near_p[256] = {0};
	unsigned int totalpixel = height * width;
	memset(his, 0, 256* sizeof(unsigned int));
	memset(near_p, 0, 256* sizeof(unsigned int));
	for (int i = 0; i < height * width; i++)
		his[(int)(data[i][0])]++;

	for (int i = 0; i < 256; i++){
		temp += his[i];
		unsigned int mt = temp << 8;
		near_p[i] = mt / totalpixel;
	}

	for (int i = 0; i < height * width; i++){
		data[i][0] = near_p[(int) data[i][0]];
		if (data[i][0] > 255){
			data[i][0] = 255;
		}
	}
}