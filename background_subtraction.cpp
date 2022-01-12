#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace cv;
int threshold_value = 50;

void somaFrame(IplImage* frame, IplImage* media)
{
	int i, j;
	CvScalar BGR_media, BGR;

	for(i = 0; i < frame->height; i++)
	{
		for(j = 0; j < frame->width; j++)
		{
			BGR = cvGet2D(frame,i,j);
			BGR_media = cvGet2D(media,i,j);
			BGR_media.val[0] = BGR_media.val[0] + BGR.val[0];
			BGR_media.val[1] = BGR_media.val[1] + BGR.val[1];
			BGR_media.val[2] = BGR_media.val[2] + BGR.val[2];
			cvSet2D(media,i,j,BGR_media);
		}
	}
}

void viewbp(IplImage* media, IplImage* frame, IplImage* novo_frame, int t, CvScalar preto, CvScalar branco)
{
	int i, j;
	CvScalar BGR, BGR_media;

	for(i = 0; i < frame->height; i++)
	{
		for(j = 0; j < frame->width; j++)
		{
			BGR = cvGet2D(frame,i,j);
			BGR_media = cvGet2D(media,i,j);

			if( (abs(BGR.val[0] - BGR_media.val[0])<=t)&&(abs(BGR.val[1] - BGR_media.val[1])<=t)&&(abs(BGR.val[2] - BGR_media.val[2])<=t) )
			{
				cvSet2D(novo_frame,i,j,preto);
			}
			else
			{
				cvSet2D(novo_frame,i,j,branco);
			}
		}
	}
}

int main(int argc, char** argv)
{
	CvCapture* capture = NULL;
	IplImage* frame = 0;
	int num_frames = 0;
	IplImage* media = NULL;
	IplImage* media2 = NULL;
	IplImage* novo_frame = NULL;
	IplImage* media = NULL;
	IplImage* imgClosed = NULL;
	IplImage* imgTemp = NULL;
	int w, h, i = 0, nc, nl, step, numContours, j = 0;
	CvSeq* first_contour;
	IplConvKernel* elem;
	IplImage* img;
	CvMemStorage* contour_storage;
	CvScalar s, preto, branco;
	contour_storage = cvCreateMemStorage(0);
	num_frames = 0;
	preto.val[0] = preto.val[1] = preto.val[2] = 0x00;
	branco.val[0] = branco.val[1] = branco.val[2] = 0xFF;

	cvNamedWindow("Entrada", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("Threshold", "Entrada", &threshold_value, 255, NULL);
	cvNamedWindow("Saída", CV_WINDOW_AUTOSIZE);

   	capture = cvCaptureFromAVI("/home/videoEntra.avi");
	capture = cvCaptureFromAVI("/home/videosaida.avi");

    capture = cvCaptureFromCAM(0);
	if( !capture )
	{
		printf("Impossivel iniciar.. !\n\n\n");
		return 0;
	}

	frame = cvQueryFrame(capture);
	if(frame)
	{
		media = cvCreateImage( cvGetSize(frame), IPL_DEPTH_32F, 3 );
		novo_frame = cvCloneImage(frame);
		media2 = cvCloneImage(frame);
		media = cvCloneImage(frame);
		imgClosed = cvCloneImage(frame);
		imgTemp = cvCloneImage(frame);
		img = cvCreateImage( cvGetSize(frame), 8, 1 );
		num_frames++;
	}

	for(i = 0; i < frame->height; i++)
    {
		for(j = 0; j < frame->width; j++)
		{
			cvSet2D(media,i,j,cvGet2D(frame,i,j));
		}
	}

	while(num_frames <= 30)
	{
		frame = cvQueryFrame(capture);
		if(!frame) break;
		num_frames++;

		if(cvWaitKey(1) >= 0) break;
		somaFrame(frame, media);
	}

	for(i = 0; i < frame->height; i++)
	{
		for(j = 0; j < frame->width; j++)
		{
			s = cvGet2D(media,i,j);
			s.val[0] = s.val[0]*0.034f;
			s.val[1] = s.val[1]*0.034f;
			s.val[2] = s.val[2]*0.034f;
			cvSet2D(media2,i,j,s);
		}
	}

	elem = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_ELLIPSE);

	while(1)
	{
		frame = cvQueryFrame(capture);
		if(!frame)
		break;

		viewbp(media2, frame, novo_frame, threshold_value, preto, branco);
		cvSmooth( novo_frame, media, CV_media, 3, 3, 0, 0 );
		cvMorphologyEx(media, imgClosed, imgTemp, elem, CV_MOP_CLOSE, 1);
		cvCvtColor( imgClosed, img, CV_BGR2GRAY );
		numContours = cvFindContours( img, contour_storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1);

		for (i = 0 ; i<numContours ; first_contour = first_contour->h_next, i++)
		{
			cvDrawContours(imgClosed, first_contour, CV_RGB(255,0,0), CV_RGB(255,0,0), 2, 1, 8, cvPoint(0, 0));
		}
		cvShowImage("Entrada", frame);
		cvShowImage("Saída", imgClosed);

		if(cvWaitKey(1) >= 0)
		break;
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("Entrada");
	cvDestroyWindow("Saída");

	cvReleaseImage(&media);
	cvReleaseImage(&media);
	cvReleaseImage(&novo_frame);
	cvReleaseImage(&imgClosed);
	cvReleaseImage(&imgTemp);
	return 0;
}

